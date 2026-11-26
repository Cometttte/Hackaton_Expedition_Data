"""Génère une tournée GETYS synthétique (.tdms) pour tester l'analyse et le dashboard.

Scénario, sur les vraies rues d'une commune (filaire OpenStreetMap, mis en cache dans dashboard/voirie/ par la
même fonction que le dashboard) :
  * un itinéraire principal (~1,2 km) parcouru N fois en alternant les sens A -> B puis B -> A,
    véhicule sur la voie de droite : les deux sens ne voient pas tout à fait le même éclairement ;
  * une boucle de détour parcourue dans UN SEUL sens (tronçons « à repasser ») ;
  * un tronçon d'environ 150 m aux lampadaires éteints (non conforme) ;
  * des lampadaires tous les 30 m en quinconce, moitié sodium (Tcp = 0 au chromamètre), moitié LED 3 000 K ;
  * du bruit : parasites sur un seul capteur, éblouissements brefs, petits offsets négatifs.

Le fichier reprend la structure des vrais relevés (groupes Luxmètres 50 Hz, GNSS port A 1 Hz avec horodatage
PC, IMU port C 5 Hz avec cap $PRDID, Calorimetre ~1 Hz) : getys_rapport.py le traite comme une vraie tournée.

    python outils/generer_tournee_test.py [--commune chasseneuil|ormesson] [-o sortie.tdms] [--passages 8]
    python outils/generer_tournee_test.py --sens1 10 --sens2 5 --date 2026-11-21
        -> 10 passages A -> B et 5 passages B -> A sur l'itinéraire principal ; pour enchaîner deux passages
           dans le même sens, le véhicule revient au départ sans enregistrement GNSS (acquisition en pause) :
           l'analyse ignore ces repositionnements, les comptes restent exacts
    ... --sens1 10 --sens2 5 --ponderations
        -> pondérations connues pour vérifier l'algorithme des tronçons (véhicule sur l'axe : mêmes conditions
           dans les deux sens) : sens 1 = passages alternés lents et lumineux (4 m/s, x1,2) / rapides et sombres
           (12 m/s, x0,8), soit x1,0 par passage (x1,1 par échantillon brut à 50 Hz, neutralisé par le pas
           de 2 m) ; sens 2 = x0,5. Attendu par tronçon :
           E_sens2 / E_sens1 = 0,50 et E_moy / E_sens1 = 0,75 (moyenne des deux sens). Vérité dans verite.json
"""
from __future__ import annotations

import argparse
import heapq
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import json
import sys

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter
from pyproj import Transformer

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "dashboard"))
import troncons as T  # noqa: E402  (filaire OSM + cache, comme le dashboard)
TO93 = Transformer.from_crs(4326, 2154, always_xy=True)
TO84 = Transformer.from_crs(2154, 4326, always_xy=True)

# extrémités de l'itinéraire principal (lat, lon) et nom de fichier (= nom de commune pour l'analyse)
COMMUNES = {
    "ormesson": dict(nom="Ormesson-sur-Marne", A=(48.7845, 2.5345), B=(48.7893, 2.5468)),
    "chasseneuil": dict(nom="Chasseneuil-du-Poitou", A=(46.6478, 0.3655), B=(46.6552, 0.3790)),
}
FS = 50.0                        # luxmètres : 50 Hz
LATERAL_CAPTEURS = {"Lux1": -0.49, "Lux2": 0.0, "Lux3": 0.49}    # lux_position du vrai TDMS (y > 0 = gauche)
VOIE = 1.6                       # le véhicule roule à 1,6 m à droite de l'axe
H_FEU, H_CAPTEUR = 7.0, 1.5      # hauteur des feux, des capteurs (remorque)
ECART_LAMPES, LATERAL_LAMPE = 30.0, 5.3
DECALAGE_HORLOGE = 3.4           # horloge PC en avance sur le GPS (s), corrigée par l'analyse


# ------------------------------------------------------------------ filaire et itinéraire
def graphe(A, B):
    """Graphe des voies OSM autour de l'itinéraire : nœuds = coordonnées partagées, arêtes = longueur (L93).
    Emprise large (~1,5 km autour de A et B) : elle couvre ensuite toute la tournée dans le dashboard."""
    lat = np.array([min(A[0], B[0]) - 0.012, max(A[0], B[0]) + 0.012])
    lon = np.array([min(A[1], B[1]) - 0.017, max(A[1], B[1]) + 0.017])
    ways = T.charger_voirie(lat, lon)
    voisins, coords = {}, {}
    for w in ways:
        g = w.get("geometry") or []
        if w.get("tags", {}).get("highway") == "service":
            continue
        cles = [(round(p["lat"], 7), round(p["lon"], 7)) for p in g]
        for k in cles:
            coords[k] = TO93.transform(k[1], k[0])
        for a, b in zip(cles[:-1], cles[1:]):
            d = math.dist(coords[a], coords[b])
            voisins.setdefault(a, []).append((b, d))
            voisins.setdefault(b, []).append((a, d))
    return voisins, coords


def plus_proche(coords, lat, lon):
    x, y = TO93.transform(lon, lat)
    return min(coords, key=lambda k: (coords[k][0] - x) ** 2 + (coords[k][1] - y) ** 2)


def dijkstra(voisins, src, dst, interdits=frozenset()):
    """Plus court chemin de src vers dst (un nœud, ou un ensemble de nœuds : le premier atteint)."""
    cibles = dst if isinstance(dst, (set, frozenset)) else {dst}
    dist, prec, file = {src: 0.0}, {}, [(0.0, src)]
    atteint = None
    while file:
        d, u = heapq.heappop(file)
        if u in cibles and u != src:
            atteint = u
            break
        if d > dist.get(u, 1e18):
            continue
        for v, w in voisins.get(u, []):
            if (u, v) in interdits or (v, u) in interdits:
                continue
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v], prec[v] = nd, u
                heapq.heappush(file, (nd, v))
    if atteint is None:
        raise RuntimeError("itinéraire introuvable")
    chemin = [atteint]
    while chemin[-1] != src:
        chemin.append(prec[chemin[-1]])
    return chemin[::-1]


def polyligne(coords, chemin):
    xy = np.array([coords[k] for k in chemin])
    return xy[np.r_[True, np.hypot(*np.diff(xy, axis=0).T) > 0.01]]


# ------------------------------------------------------------------ cinématique
def rouler(xy, t0, v_croisiere, rng, arret_fin=6.0, variation=True):
    """Échantillonne un trajet à 50 Hz : positions de l'axe, cap, vitesse (accélération / freinage)."""
    s_pts = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    L = s_pts[-1]
    v_max = v_croisiere * (rng.uniform(0.85, 1.15) if variation else 1.0)
    acc = 1.5
    t_acc = v_max / acc
    d_acc = 0.5 * acc * t_acc ** 2
    if 2 * d_acc > L:
        d_acc, v_max = L / 2, math.sqrt(acc * L)
        t_acc = v_max / acc
    t_cr = (L - 2 * d_acc) / v_max
    T = 2 * t_acc + t_cr
    t = np.arange(0, T + arret_fin, 1 / FS)
    s = np.where(t < t_acc, 0.5 * acc * t ** 2,
                 np.where(t < t_acc + t_cr, d_acc + v_max * (t - t_acc),
                          np.where(t < T, L - 0.5 * acc * (T - t) ** 2, L)))
    s = np.clip(s, 0, L)
    x, y = np.interp(s, s_pts, xy[:, 0]), np.interp(s, s_pts, xy[:, 1])
    seg = np.clip(np.searchsorted(s_pts, s, side="right") - 1, 0, len(xy) - 2)
    cap = np.degrees(np.arctan2(xy[seg + 1, 0] - xy[seg, 0], xy[seg + 1, 1] - xy[seg, 1])) % 360
    v = np.gradient(s) * FS
    return t0 + t, x, y, cap, v


# ------------------------------------------------------------------ lampadaires et éclairement
def lampadaires(xy, rng, eteints=None, debut_type=0):
    """Lampadaires tous les 30 m en quinconce le long d'une polyligne : (x, y, intensité cd, type)."""
    s_pts = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    out = []
    for k, s in enumerate(np.arange(10, s_pts[-1] - 5, ECART_LAMPES)):
        i = min(np.searchsorted(s_pts, s, side="right") - 1, len(xy) - 2)
        dx, dy = xy[i + 1] - xy[i]
        n = math.hypot(dx, dy) or 1
        gx, gy = -dy / n, dx / n                          # gauche de la polyligne
        cote = 1 if k % 2 == 0 else -1
        px, py = np.interp(s, s_pts, xy[:, 0]) + cote * LATERAL_LAMPE * gx, \
            np.interp(s, s_pts, xy[:, 1]) + cote * LATERAL_LAMPE * gy
        typ = "sodium" if ((k // 12) + debut_type) % 2 == 0 else "led_3000"
        I = rng.uniform(1300, 1900) if typ == "sodium" else rng.uniform(1000, 1500)
        if eteints is not None and eteints[0] <= s <= eteints[1]:
            I = 0.0                                       # lampadaire éteint
        out.append((px, py, I, typ))
    return out


def eclairement(px, py, lamps):
    """Éclairement horizontal (loi du cosinus) en chaque point capteur : E = I · h / d³, + fond."""
    lx = np.array([l[0] for l in lamps])
    ly = np.array([l[1] for l in lamps])
    li = np.array([l[2] for l in lamps])
    dh = H_FEU - H_CAPTEUR
    E = np.full(px.shape, 0.6)
    for k in range(0, len(px), 20000):
        sl = slice(k, k + 20000)
        d2 = (px[sl, None] - lx[None, :]) ** 2 + (py[sl, None] - ly[None, :]) ** 2 + dh ** 2
        E[sl] += (li[None, :] * dh / d2 ** 1.5).sum(axis=1)
    return E


# ------------------------------------------------------------------ NMEA
def cs(phrase):
    c = 0
    for ch in phrase:
        c ^= ord(ch)
    return f"*{c:02X}"


def ddmm(v, lat):
    d = int(abs(v))
    m = (abs(v) - d) * 60
    return (f"{d:02d}{m:010.7f}" if lat else f"{d:03d}{m:010.7f}"), ("N" if v >= 0 else "S") if lat else ("E" if v >= 0 else "W")


def gga(t, lat, lon):
    h = datetime.fromtimestamp(t, timezone.utc)
    la, ns = ddmm(lat, True)
    lo, ew = ddmm(lon, False)
    p = f"GPGGA,{h:%H%M%S}.{int(h.microsecond / 1e4):02d},{la},{ns},{lo},{ew},2,14,0.8,98.000,M,46.785,M,2.0,0136"
    return f"${p}{cs(p)}"


def zda(t):
    h = datetime.fromtimestamp(t, timezone.utc)
    p = f"GPZDA,{h:%H%M%S}.{int(h.microsecond / 1e4):02d},{h:%d},{h:%m},{h:%Y},,"
    return f"${p}{cs(p)}"


def prdid(cap):
    p = f"PRDID,+000.40,-000.90,{cap:05.1f}"
    return f"${p}{cs(p)}"


# ------------------------------------------------------------------ génération
def generer(sortie: Path, n_passages: int, graine: int, A, B, sens1=None, sens2=None, jour="2026-11-20",
            ponderations=False):
    rng = np.random.default_rng(graine)
    voisins, coords = graphe(A, B)
    na, nb = plus_proche(coords, *A), plus_proche(coords, *B)
    principal = dijkstra(voisins, na, nb)
    xy_p = polyligne(coords, principal)
    L = float(np.hypot(*np.diff(xy_p, axis=0).T).sum())
    # boucle de détour : depuis un nœud à ~40 % de l'itinéraire, retour au même nœud par d'autres rues
    dans = set(principal)
    carrefours = [i for i in range(len(principal) // 5, 3 * len(principal) // 5)
                  if any(v not in dans for v, _ in voisins[principal[i]])]       # nœuds avec une rue latérale
    if not carrefours:
        raise RuntimeError("pas de carrefour pour le détour")
    i_det = min(carrefours, key=lambda i: abs(i - len(principal) * 2 // 5))
    n_det = principal[i_det]
    interdits = frozenset(zip(principal[:-1], principal[1:]))
    # détour : aller vers un nœud à ~160 m par une rue latérale, puis retour sur l'itinéraire principal plus loin
    # (premier carrefour atteint après le départ du détour), sans repasser par les mêmes rues
    suite = frozenset(principal[i_det + 1:])
    candidats = sorted((k for k in coords if k not in dans and 90 < math.dist(coords[k], coords[n_det]) < 300),
                       key=lambda k: abs(math.dist(coords[k], coords[n_det]) - 160))
    boucle = None
    for loin in candidats[:400]:
        try:
            aller = dijkstra(voisins, n_det, loin, interdits)
            retour = dijkstra(voisins, loin, suite, frozenset(zip(aller[:-1], aller[1:])) | interdits)
        except RuntimeError:
            continue
        if not set(aller[1:-1]) & set(retour[1:-1]) and not set(aller[1:]) & dans:
            boucle = aller + retour[1:]
            break
    if boucle is None:
        raise RuntimeError("pas de détour trouvé")
    j_ret = principal.index(boucle[-1])                           # où le détour rejoint l'itinéraire
    xy_b = polyligne(coords, boucle)

    # --- programme des trajets
    par_sens = sens1 is not None and sens2 is not None
    xy_alt = None
    if par_sens:
        # N1 passages A -> B, N2 passages B -> A sur l'itinéraire principal, en alternant tant que possible ;
        # pour enchaîner deux passages dans le même sens : repositionnement sans GNSS (« transit »)
        n_min = min(sens1, sens2)
        plan = [("principal", True), ("principal", False)] * n_min
        if sens1 > sens2:
            plan += [("principal", True), ("transit", None)] * (sens1 - n_min)
        elif sens2 > sens1:
            plan += [("transit", None), ("principal", False)] * (sens2 - n_min)
        lamps = lampadaires(xy_p, rng, eteints=(0.62 * L, 0.62 * L + 150))
    else:
        # A -> B, B -> A, ... ; détour (même sens) pendant les 2 premiers allers
        plan = [("detour", None) if (k % 2 == 0 and k < 4) else ("principal", k % 2 == 0) for k in range(n_passages)]
        lamps = lampadaires(xy_p, rng, eteints=(0.62 * L, 0.62 * L + 150)) + lampadaires(xy_b, rng, debut_type=1)

    a, m_, j = (int(x) for x in jour.split("-"))
    t_gps0 = datetime(a, m_, j, 19, 0, 0, tzinfo=timezone.utc).timestamp()   # 20h00 heure locale
    morceaux, facteurs, t = [], [], t_gps0
    PONDS = {"lent": (4.0, 1.2), "rapide": (12.0, 0.8), "sens2": (8.0, 0.5)}    # vitesse m/s, facteur d'éclairement
    n_aller, programme = 0, []
    for quoi, aller in plan:
        if quoi == "detour":
            pre = polyligne(coords, principal[: i_det + 1])          # A -> départ du détour
            post = polyligne(coords, principal[j_ret:])              # retour sur l'itinéraire -> B
            trajet = np.vstack([pre, xy_b[1:], post[1:]])
        elif quoi == "transit":                                  # repositionnement : pas de position GNSS
            tt = t + np.arange(0, L / 9.0, 1 / FS)
            nan = np.full(tt.size, np.nan)
            morceaux.append((tt, nan, nan, nan, nan))
            facteurs.append(np.ones(tt.size))
            t = tt[-1] + 1 / FS
            continue
        else:
            trajet = xy_p if aller else xy_p[::-1]
        vitesse, facteur, variation = rng.uniform(7.5, 10.0), 1.0, True
        if ponderations and quoi == "principal":
            cle = ("lent" if n_aller % 2 == 0 else "rapide") if aller else "sens2"
            n_aller += int(bool(aller))
            (vitesse, facteur), variation = PONDS[cle], False
            programme.append(dict(sens=1 if aller else 2, type=cle, vitesse_ms=vitesse, facteur=facteur))
        r = rouler(trajet, t, vitesse, rng, variation=variation)
        morceaux.append(r)
        facteurs.append(np.full(r[0].size, facteur))
        t = r[0][-1] + 1 / FS
    t_all = np.concatenate([m[0] for m in morceaux])
    x = np.concatenate([m[1] for m in morceaux])
    y = np.concatenate([m[2] for m in morceaux])
    cap = np.concatenate([m[3] for m in morceaux])
    v = np.concatenate([m[4] for m in morceaux])
    rad = np.radians(cap)
    gx, gy = -np.cos(rad), np.sin(rad)                    # vecteur « gauche » du véhicule
    voie = 0.0 if ponderations else VOIE                  # pondérations : sur l'axe, mêmes conditions dans les 2 sens
    xv, yv = x - voie * gx, y - voie * gy                 # voie de droite
    facteur = np.concatenate(facteurs)

    # --- luxmètres : chaque capteur à sa position latérale, bruit et aberrations
    L3 = {}
    for nom, dl in LATERAL_CAPTEURS.items():
        sur = np.isfinite(xv)
        E = np.full(xv.shape, 0.6)
        E[sur] = eclairement((xv + dl * gx)[sur], (yv + dl * gy)[sur], lamps)
        E = E * facteur                                   # pondération du passage (1 hors scénario de test)
        E = E * rng.normal(1.0, 0.025, E.size) + rng.normal(0, 0.15, E.size) - 0.25   # gain, bruit, offset
        L3[nom] = E
    n = t_all.size
    for _ in range(int(n / FS / 25)):                     # parasites sur un seul capteur (filtre de Hampel)
        i = rng.integers(0, n - 5)
        L3[rng.choice(list(LATERAL_CAPTEURS))][i:i + 3] += rng.uniform(60, 220)
    for _ in range(int(n / FS / 70)):                     # éblouissements brefs sur les 3 capteurs (« trop bref »)
        i = rng.integers(0, n - 10)
        for nom in L3:
            L3[nom][i:i + 6] += rng.uniform(90, 160)

    # --- GNSS : positions de l'antenne (axe du véhicule) + bruit, port C 5 Hz et port A 1 Hz
    lon, lat = TO84.transform(xv + rng.normal(0, 0.25, n), yv + rng.normal(0, 0.25, n))
    gnss_ok = np.isfinite(xv)
    idx_c = np.arange(0, n, int(FS / 5)); idx_c = idx_c[gnss_ok[idx_c]]
    idx_a = np.arange(0, n, int(FS)); idx_a = idx_a[gnss_ok[idx_a]]
    pc = lambda tt: tt + DECALAGE_HORLOGE + rng.uniform(0.02, 0.08)   # noqa: E731  (réception PC)
    octets, dates_c = [], []
    for i in idx_c:
        bloc = "\r\n".join([prdid(cap[i]), gga(t_all[i], lat[i], lon[i]), zda(t_all[i])]) + "\r\n"
        octets.append(np.frombuffer(bloc.encode("ascii"), np.uint8))
        dates_c.append(pc(t_all[i]))
    msgs_a, dates_a = [], []
    for i in idx_a:
        for m in (gga(t_all[i], lat[i], lon[i]), zda(t_all[i])):
            msgs_a.append(m + "\r\n")
            dates_a.append(pc(t_all[i]))

    # --- chromamètre ~1 Hz : Tcp = 0 sous sodium, ~3 000 K sous LED
    lxs = np.array([l[0] for l in lamps]); lys = np.array([l[1] for l in lamps])
    typ = np.array([l[3] for l in lamps]); actif = np.array([l[2] > 0 for l in lamps])
    idx_cal = np.arange(0, n, int(FS * 1.05))
    tcp = []
    for i in idx_cal:
        if not gnss_ok[i]:
            tcp.append(0.0)
            continue
        d = np.hypot(lxs - xv[i], lys - yv[i]) + np.where(actif, 0, 1e9)
        j = int(np.argmin(d))
        tcp.append(0.0 if (d[j] > 25 or typ[j] == "sodium") else float(rng.normal(2990, 40)))

    # --- écriture : horodatages PC (UTC naïf), comme les vrais relevés
    def dt64(ts):
        return np.array([np.datetime64(int(round(s * 1e6)), "us") for s in np.atleast_1d(ts)])
    t_pc0 = t_all[0] + DECALAGE_HORLOGE
    props_lux = lambda nom, dl: {"wf_start_time": dt64(t_pc0)[0], "wf_start_offset": 0.0,   # noqa: E731
                                 "wf_increment": 1 / FS, "wf_samples": 1, "NI_UnitDescription": "Lux",
                                 "unit_string": "Lux", "lux_offset": 0.0,
                                 "lux_position": f"[{0.25 if dl else 0.18:.6f};{dl:.6f}]"}
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with TdmsWriter(str(sortie)) as w:
        w.write_segment([
            RootObject({"name": datetime.fromtimestamp(t_pc0, timezone.utc).strftime("%Y%m%d%H%M%S"),
                        "description": "Tournée synthétique de test (outils/generer_tournee_test.py)"}),
            GroupObject("Luxmètres"),
            *[ChannelObject("Luxmètres", nom, L3[nom], properties=props_lux(nom, dl))
              for nom, dl in LATERAL_CAPTEURS.items()],
            GroupObject("GNSS"),
            ChannelObject("GNSS", "date Port A", dt64(dates_a)),
            ChannelObject("GNSS", "Port A", np.array(msgs_a)),
            ChannelObject("GNSS", "taille Port A", np.array([len(m) for m in msgs_a], np.int32)),
            GroupObject("IMU"),
            ChannelObject("IMU", "date Port C", dt64(dates_c)),
            ChannelObject("IMU", "Port C", np.concatenate(octets)),
            ChannelObject("IMU", "taille Port C", np.array([o.size for o in octets], np.int32)),
            GroupObject("Calorimetre"),
            ChannelObject("Calorimetre", "Date", dt64(t_all[idx_cal] + DECALAGE_HORLOGE)),
            ChannelObject("Calorimetre", "Ev", np.round(L3["Lux2"][idx_cal], 1)),
            ChannelObject("Calorimetre", "Tcp", np.array(tcp)),
            ChannelObject("Calorimetre", "deltaUV", np.where(np.array(tcp) > 0, 0.0015, 0.0)),
        ])
    if ponderations:
        verite = dict(
            scenario="pondérations connues (véhicule sur l'axe : mêmes conditions géométriques dans les deux sens)",
            passages=programme,
            attendu=dict(
                sens1_par_passage=1.0,
                sens1_par_echantillon_brut_50Hz=round((1.2 * 3 + 0.8 * 1) / 4, 3),  # neutralisé par le pas de 2 m
                sens2=0.5, ratio_sens2_sur_sens1=0.5, ratio_Emoy_sur_sens1_moyenne_des_sens=0.75,
                ratio_si_ponderation_par_nombre_de_passages=round((sens1 * 1.0 + sens2 * 0.5) / (sens1 + sens2), 3)),
            note=("sens 1 = sens de dessin du tronçon OSM ou l'inverse : comparer max(E_sens) et min(E_sens). "
                  "Les passages lents produisent 3x plus d'échantillons à 50 Hz, mais le ré-échantillonnage tous "
                  "les 2 m de getys_analyse.py donne autant de points par mètre à chaque passage : attendu x1,0."))
        (sortie.parent / "verite.json").write_text(json.dumps(verite, ensure_ascii=False, indent=2), encoding="utf-8")
    duree = (t_all[-1] - t_all[0]) / 60
    print(f"✓ {sortie}  ({sortie.stat().st_size / 1e6:.1f} Mo)")
    if par_sens:
        print(f"  itinéraire principal {L:.0f} m : {sens1} passages A -> B, {sens2} passages B -> A"
              + (f" ; {abs(sens1 - sens2)} repositionnements sans GNSS" if sens1 != sens2 else ""))
    else:
        print(f"  itinéraire principal {L:.0f} m, {n_passages} passages alternés ; détour "
          f"{float(np.hypot(*np.diff(xy_b, axis=0).T).sum()):.0f} m, 2 passages dans un seul sens")
    print(f"  {len(lamps)} lampadaires (dont {int((~actif).sum())} éteints sur ~150 m), "
          f"{duree:.0f} min de relevé, {n} échantillons par capteur")


def main():
    ap = argparse.ArgumentParser(description="Tournée GETYS synthétique pour tester le dashboard")
    ap.add_argument("--commune", choices=sorted(COMMUNES), default="chasseneuil")
    ap.add_argument("-o", "--sortie", default=None, help="défaut : donnees_test/<Commune>.tdms")
    ap.add_argument("--passages", type=int, default=8, help="passages sur l'itinéraire principal (alternés)")
    ap.add_argument("--graine", type=int, default=1, help="graine du hasard (résultat reproductible)")
    ap.add_argument("--sens1", type=int, default=None, help="passages A -> B (avec --sens2 : remplace --passages)")
    ap.add_argument("--sens2", type=int, default=None, help="passages B -> A")
    ap.add_argument("--date", default="2026-11-20", help="date de la tournée (AAAA-MM-JJ), 20h00 heure locale")
    ap.add_argument("--ponderations", action="store_true",
                    help="avec --sens1/--sens2 : pondérations connues pour vérifier l'algorithme des tronçons")
    a = ap.parse_args()
    c = COMMUNES[a.commune]
    sortie = Path(a.sortie) if a.sortie else RACINE / "donnees_test" / f"{c['nom']}.tdms"
    generer(sortie, a.passages, a.graine, c["A"], c["B"], a.sens1, a.sens2, a.date, a.ponderations)


if __name__ == "__main__":
    main()
