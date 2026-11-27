"""Moyenne au tronçon sur le filaire de voirie (OpenStreetMap).

1. Filaire : voies carrossables OSM de l'emprise des mesures (Overpass, mis en cache dans voirie/).
2. Tronçons : un tronçon va d'un croisement au croisement suivant, et ne se coupe que là
   (pas de recoupage par longueur). Croisement = nœud OSM où arrivent au moins 3 segments de voie
   (carrefour, entrée de rond-point, raccord de bretelle). Deux ways OSM qui se suivent bout à bout
   sans croisement (rue découpée pour un pont, etc.) sont fusionnées ; on coupe seulement si le nom,
   le type de voie ou le sens unique y change. Rond-point : un tronçon entre deux entrées.
3. Map-matching : chaque mesure (pas de 2 m) est rattachée au tronçon le plus proche (< 15 m)
   dont l'orientation est cohérente avec le cap du véhicule (évite les erreurs aux carrefours).
4. Multi-passages : un passage = suite continue de mesures sur un même tronçon (coupure > 20 s).
   Chaque passage est classé selon son sens de circulation (cap du véhicule vs sens du tronçon).
5. Moyenne par sens = moyenne des moyennes de passage de ce sens (chaque passage pèse autant,
   quelle que soit la vitesse). E_moy du tronçon = moyenne des deux sens -> les deux côtés de la
   route comptent autant, même si un sens a été parcouru plus souvent. Un seul sens mesuré : E_moy
   = ce sens, et le tronçon est signalé (n_sens = 1).
6. Seuls les tronçons suffisamment parcourus sont gardés (couverture >= 30 % de leur longueur).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree

VOIRIE_DIR = Path(__file__).resolve().parent / "voirie"   # cache du filaire OSM
OVERPASS = ["https://overpass-api.de/api/interpreter",                     # serveurs publics, essayés dans l'ordre
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
TYPES = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|road|"
         "motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")

# Classes SDAL (hypothèses par défaut, modifiables dans le dashboard)
CLASSES = {
    "principale": ("Voie principale", ("motorway", "trunk", "primary", "secondary"), 15.0),
    "distribution": ("Voie de distribution", ("tertiary", "unclassified", "road"), 10.0),
    "residentielle": ("Voie résidentielle", ("residential", "living_street"), 7.5),
    "desserte": ("Desserte / service", ("service",), 5.0),
}
CLASSE_DE = {h: c for c, (_, hs, _) in CLASSES.items() for h in hs}

RAYON_M = 15.0        # distance max mesure -> axe de voie (capteurs décalés + précision GNSS)
ANGLE_MAX = 40.0      # écart max cap véhicule / orientation du tronçon (degrés, modulo 180)
GAP_S = 20.0          # deux mesures d'un même tronçon séparées de plus de 20 s = 2 passages
PAS_M = 2.0           # densification des tronçons
COUV_MIN = 0.30

_to93 = Transformer.from_crs(4326, 2154, always_xy=True)
_to84 = Transformer.from_crs(2154, 4326, always_xy=True)


def classe(highway: str) -> str:
    return CLASSE_DE.get(highway.replace("_link", "") if highway.endswith("_link") else highway, "desserte")


# ------------------------------------------------------------------ FILAIRE OSM
def _bbox(lat, lon, marge=0.0005):
    """Emprise des mesures + marge, arrondie vers l'extérieur (1e-4 °)."""
    return (math.floor((lat.min() - marge) * 1e4) / 1e4, math.floor((lon.min() - marge) * 1e4) / 1e4,
            math.ceil((lat.max() + marge) * 1e4) / 1e4, math.ceil((lon.max() + marge) * 1e4) / 1e4)


def charger_voirie(lat, lon) -> list[dict]:
    """Voies OSM couvrant les mesures. Cache disque : un fichier par emprise."""
    s, w, n, e = _bbox(lat, lon)
    VOIRIE_DIR.mkdir(exist_ok=True)
    for f in VOIRIE_DIR.glob("osm_*.json"):          # une emprise en cache qui contient la nôtre
        try:
            cs, cw, cn, ce = map(float, f.stem.split("_")[1:5])
        except ValueError:
            continue
        if cs <= s and cw <= w and cn >= n and ce >= e:
            return json.loads(f.read_text(encoding="utf-8"))["elements"]
    q = f'[out:json][timeout:120];way["highway"~"^({TYPES})$"]({s},{w},{n},{e});out geom tags;'
    derniere = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, timeout=90,
                              headers={"User-Agent": "getys-dashboard/1.0"})
            r.raise_for_status()
            d = r.json()
            (VOIRIE_DIR / f"osm_{s}_{w}_{n}_{e}.json").write_text(json.dumps(d), encoding="utf-8")
            return d["elements"]
        except (requests.RequestException, ValueError) as ex:
            derniere = ex
    raise RuntimeError(f"Filaire OSM indisponible : {derniere}")


def decouper(ways: list[dict]) -> list[dict]:
    """Tronçons de croisement à croisement (coordonnées Lambert 93).

    Le filaire est vu comme un graphe : nœud OSM = sommet, segment de voie = arête, degré d'un nœud =
    nombre de segments qui y arrivent. Degré >= 3 : croisement (carrefour, entrée de rond-point,
    raccord de bretelle) -> coupure. Degré 1 : impasse. Degré 2 : simple point de passage, y compris
    quand deux ways OSM se suivent bout à bout (rue découpée pour un pont, un changement de
    revêtement ou de limitation...) -> pas de coupure, les deux ways sont fusionnées. Exception : si le
    nom, le type de voie ou le sens unique change à ce nœud, on coupe quand même (un tronçon porte une
    seule classe SDAL et un seul nom).
    """
    # nœud partagé = mêmes coordonnées dans plusieurs voies (OSM : un nœud = un point unique)
    cle = lambda g: (round(g["lat"], 7), round(g["lon"], 7))   # noqa: E731
    voies, degre, coord = [], Counter(), {}
    for w in ways:
        geom = [g for g in w.get("geometry", []) if g]
        nodes = []
        for g in geom:
            n = cle(g)
            coord.setdefault(n, (g["lon"], g["lat"]))
            if not nodes or nodes[-1] != n:             # doublons consécutifs ignorés
                nodes.append(n)
        if len(nodes) < 2:
            continue
        for a, b in zip(nodes[:-1], nodes[1:]):
            degre[a] += 1
            degre[b] += 1
        tags = w.get("tags", {})
        hw = tags.get("highway", "road")
        attrs = {"nom": tags.get("name") or tags.get("ref") or "", "highway": hw, "classe": classe(hw),
                 "sens_unique": tags.get("oneway") in ("yes", "1", "-1") or tags.get("junction") == "roundabout"}
        voies.append((w["id"], nodes, attrs))

    # 1. morceaux de way entre deux croisements (coupure aux nœuds intérieurs de degré >= 3)
    morceaux = []                                       # (id, nœuds, attributs)
    for wid, nodes, attrs in voies:
        coupe = [0] + [i for i in range(1, len(nodes) - 1) if degre[nodes[i]] >= 3] + [len(nodes) - 1]
        for a, b in zip(coupe[:-1], coupe[1:]):
            morceaux.append((f"{wid}_{a}", nodes[a:b + 1], attrs))

    # 2. continuité : nœud de degré 2 où se touchent exactement deux extrémités de morceaux différents
    #    de mêmes nom / type / sens unique -> les deux morceaux sont chaînés
    bouts = {}
    for k, (_, nodes, _) in enumerate(morceaux):
        if nodes[0] != nodes[-1]:                       # un morceau fermé (boucle) n'a pas de bout libre
            bouts.setdefault(nodes[0], []).append(k)
            bouts.setdefault(nodes[-1], []).append(k)
    attrs_fusion = lambda k: tuple(morceaux[k][2][c] for c in ("nom", "highway", "sens_unique"))  # noqa: E731
    voisin = {}                                         # (morceau, nœud d'extrémité) -> morceau suivant
    for n, ks in bouts.items():
        if degre[n] == 2 and len(ks) == 2 and ks[0] != ks[1] and attrs_fusion(ks[0]) == attrs_fusion(ks[1]):
            voisin[(ks[0], n)], voisin[(ks[1], n)] = ks[1], ks[0]

    vus, chaines = set(), []

    def suivre(k, nodes):                               # prolonge la chaîne depuis son dernier nœud
        while (k, nodes[-1]) in voisin and voisin[(k, nodes[-1])] not in vus:
            k = voisin[(k, nodes[-1])]
            n2 = morceaux[k][1]
            nodes = nodes + (n2 if n2[0] == nodes[-1] else n2[::-1])[1:]
            vus.add(k)
        return nodes

    # départ sur un bout libre (croisement, impasse, changement de rue) ; ensuite les boucles restantes
    milieu = {k for k in range(len(morceaux))
              if (k, morceaux[k][1][0]) in voisin and (k, morceaux[k][1][-1]) in voisin}
    for k in [k for k in range(len(morceaux)) if k not in milieu] + sorted(milieu):
        if k in vus:
            continue
        vus.add(k)
        nodes = morceaux[k][1]
        if (k, nodes[0]) in voisin and (k, nodes[-1]) not in voisin:
            nodes = nodes[::-1]                         # le bout chaîné passe en fin de liste
        chaines.append((morceaux[k][0], suivre(k, nodes), morceaux[k][2]))

    # 3. géométrie Lambert 93
    tr = []
    for tid, nodes, attrs in chaines:
        lon = [coord[n][0] for n in nodes]
        lat = [coord[n][1] for n in nodes]
        x, y = _to93.transform(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        lg = float(np.hypot(np.diff(x), np.diff(y)).sum())
        if lg < 4:                                      # miette de carrefour
            continue
        tr.append({**attrs, "id": tid, "lon": lon, "lat": lat, "x": x, "y": y, "longueur_m": lg})
    return tr


def _densifier(tr):
    """Points tous les 2 m le long des tronçons, avec l'orientation locale (0-180°)."""
    xs, ys, ids, angs, abs_ = [], [], [], [], []
    for k, t in enumerate(tr):
        x, y = t["x"], t["y"]
        d = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        s = np.arange(0, d[-1] + 1e-9, PAS_M)
        seg = np.clip(np.searchsorted(d, s, side="right") - 1, 0, len(x) - 2)
        ang = np.degrees(np.arctan2(x[seg + 1] - x[seg], y[seg + 1] - y[seg])) % 360   # cap du tronçon
        xs.append(np.interp(s, d, x)); ys.append(np.interp(s, d, y))
        ids.append(np.full(len(s), k)); angs.append(ang); abs_.append(s)
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(ids), np.concatenate(angs),
            np.concatenate(abs_))


# ------------------------------------------------------------------ CALCUL
def calculer(m: pd.DataFrame, ways: list[dict]):
    """m : mesures (colonnes x, y en L93, lux, cap, heure_locale).
    Retourne (tronçons DataFrame, géométries {id: (lon, lat)}, tronçon de chaque mesure)."""
    tr = decouper(ways)
    sx, sy, sid, sang, sabs = _densifier(tr)
    arbre = cKDTree(np.c_[sx, sy])

    px, py = m["x"].to_numpy(), m["y"].to_numpy()
    cap360 = (m["cap"].to_numpy() if "cap" in m else np.full(len(m), np.nan)) % 360
    cap = cap360 % 180
    dist, idx = arbre.query(np.c_[px, py], k=12, distance_upper_bound=RAYON_M)
    valide = np.isfinite(dist)
    idx_s = np.where(valide, idx, 0)
    ecart = np.abs(sang[idx_s] % 180 - cap[:, None])
    ecart = np.minimum(ecart, 180 - ecart)
    score = np.where(valide, dist + np.where(np.isnan(ecart), 0, np.where(ecart > ANGLE_MAX, 1e3, ecart / 4)), np.inf)
    best = score.argmin(axis=1)
    ok = np.isfinite(score[np.arange(len(m)), best]) & (score[np.arange(len(m)), best] < 1e3)
    echant = idx_s[np.arange(len(m)), best]
    tid = np.where(ok, sid[echant], -1)
    # sens de circulation : 1 = dans le sens de dessin du tronçon, 2 = sens inverse
    sens = np.where(np.cos(np.radians(cap360 - sang[echant])) >= 0, 1, 2)
    sens = np.where(np.isnan(cap360), 1, sens)

    # lissage : un tronçon vu sur moins de 3 points consécutifs (6 m) est un artefact de carrefour
    run = np.r_[0, np.cumsum(tid[1:] != tid[:-1])]
    taille = np.bincount(run)[run]
    tid = np.where((taille < 3) & (tid >= 0), -1, tid)

    df = pd.DataFrame({"t": tid, "sens": sens, "lux": m["lux"].to_numpy(), "h": m["heure_locale"].to_numpy() * 3600,
                       "bin": np.where(ok, (sabs[echant] // 10).astype(int), -1)})
    df = df[df["t"] >= 0]
    t_, h_ = df["t"].to_numpy(), df["h"].to_numpy()
    newp = np.r_[True, (t_[1:] != t_[:-1]) | (np.diff(h_) > GAP_S)]
    df["passage"] = np.cumsum(newp)

    # un passage = un sens (sens majoritaire de ses points)
    pas = df.groupby(["t", "passage"]).agg(Ep=("lux", "mean"), sens=("sens", lambda s: s.mode().iat[0]))
    pas = pas.reset_index()
    par_sens = pas.groupby(["t", "sens"])["Ep"].agg(["mean", "size"]).unstack("sens")
    e1 = par_sens["mean"].get(1, pd.Series(np.nan, index=par_sens.index))
    e2 = par_sens["mean"].get(2, pd.Series(np.nan, index=par_sens.index))
    n1 = par_sens["size"].get(1, pd.Series(0, index=par_sens.index)).fillna(0)
    n2 = par_sens["size"].get(2, pd.Series(0, index=par_sens.index)).fillna(0)
    gp = pas.groupby("t")["Ep"]
    gpts = df.groupby("t")["lux"]
    stats = pd.DataFrame({
        "n_passages": gp.size(),
        "E_moy": pd.concat([e1, e2], axis=1).mean(axis=1),        # moyenne des deux sens
        "E_sens1": e1, "E_sens2": e2, "n_pass_sens1": n1, "n_pass_sens2": n2,
        "n_sens": e1.notna().astype(int) + e2.notna().astype(int),
        "cv_passages": (gp.std() / gp.mean()).fillna(0),
        "E_med": gpts.median(), "E_p10": gpts.quantile(0.10), "E_min": gpts.quantile(0.05),
        "E_max": gpts.max(), "n_pts": gpts.size(), "bins": df.groupby("t")["bin"].nunique(),
    })
    stats["U0"] = (stats["E_min"] / stats["E_moy"]).where(stats["E_moy"] > 0, 0)

    lignes, geoms = [], {}
    for k, s in stats.iterrows():
        t = tr[int(k)]
        couv = min(1.0, s["bins"] / max(1, math.ceil(t["longueur_m"] / 10)))
        if couv < COUV_MIN or s["n_pts"] < 4:
            continue
        nom = t["nom"] or f"Voie sans nom ({t['highway']})"
        lignes.append({"id": t["id"], "nom": nom, "highway": t["highway"], "classe": t["classe"],
                       "sens_unique": bool(t["sens_unique"]),
                       "longueur_m": round(t["longueur_m"], 1), "couverture": round(100 * couv),
                       "n_passages": int(s["n_passages"]), "n_pts": int(s["n_pts"]),
                       "n_sens": int(s["n_sens"]),
                       "E_sens1": None if pd.isna(s["E_sens1"]) else round(s["E_sens1"], 2),
                       "E_sens2": None if pd.isna(s["E_sens2"]) else round(s["E_sens2"], 2),
                       "n_pass_sens1": int(s["n_pass_sens1"]), "n_pass_sens2": int(s["n_pass_sens2"]),
                       "E_moy": round(s["E_moy"], 2), "E_med": round(s["E_med"], 2),
                       "E_p10": round(s["E_p10"], 2), "E_max": round(s["E_max"], 1),
                       "U0": round(s["U0"], 2), "cv_passages": round(100 * s["cv_passages"])})
        geoms[t["id"]] = (t["lon"], t["lat"])
    res = pd.DataFrame(lignes)
    pt_troncon = pd.Series([tr[i]["id"] if i >= 0 else None for i in tid], index=m.index)
    return res, geoms, pt_troncon


def sauver(dossier: Path, res: pd.DataFrame, geoms: dict):
    feats = [{"type": "Feature", "properties": r,
              "geometry": {"type": "LineString", "coordinates": [list(c) for c in zip(*geoms[r["id"]])]}}
             for r in res.to_dict("records")]
    (dossier / "troncons.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False), encoding="utf-8")


def lire(dossier: Path):
    gj = json.loads((dossier / "troncons.geojson").read_text(encoding="utf-8"))
    res = pd.DataFrame([f["properties"] for f in gj["features"]])
    geoms = {f["properties"]["id"]: tuple(zip(*f["geometry"]["coordinates"])) for f in gj["features"]}
    return res, geoms
