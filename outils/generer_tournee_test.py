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
"""
from __future__ import annotations

import argparse
import heapq
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
def rouler(xy, t0, v_croisiere, rng, arret_fin=6.0):
    """Échantillonne un trajet à 50 Hz : positions de l'axe, cap, vitesse (accélération / freinage)."""
    s_pts = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    L = s_pts[-1]
    v_max = v_croisiere * rng.uniform(0.85, 1.15)
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
def generer(sortie: Path, n_passages: int, graine: int, A, B):
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

    lamps = lampadaires(xy_p, rng, eteints=(0.62 * L, 0.62 * L + 150)) + lampadaires(xy_b, rng, debut_type=1)

    # --- trajets : A->B, B->A, ... ; détour (même sens) pendant les 2 premiers allers
    t_gps0 = datetime(2026, 11, 20, 19, 0, 0, tzinfo=timezone.utc).timestamp()   # 20h00 heure locale
    morceaux, t = [], t_gps0
    for k in range(n_passages):
        if k % 2 == 0 and k < 4:
            pre = polyligne(coords, principal[: i_det + 1])          # A -> départ du détour
            post = polyligne(coords, principal[j_ret:])              # retour sur l'itinéraire -> B
            trajet = np.vstack([pre, xy_b[1:], post[1:]])
        else:
            trajet = xy_p if k % 2 == 0 else xy_p[::-1]
        r = rouler(trajet, t, rng.uniform(7.5, 10.0), rng)
        morceaux.append(r)
        t = r[0][-1] + 1 / FS
    t_all = np.concatenate([m[0] for m in morceaux])
    x = np.concatenate([m[1] for m in morceaux])
    y = np.concatenate([m[2] for m in morceaux])
    cap = np.concatenate([m[3] for m in morceaux])
    v = np.concatenate([m[4] for m in morceaux])
    rad = np.radians(cap)
    gx, gy = -np.cos(rad), np.sin(rad)                    # vecteur « gauche » du véhicule
    xv, yv = x - VOIE * gx, y - VOIE * gy                 # voie de droite

    # --- luxmètres : chaque capteur à sa position latérale, bruit et aberrations
    L3 = {}
    for nom, dl in LATERAL_CAPTEURS.items():
        E = eclairement(xv + dl * gx, yv + dl * gy, lamps)
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
    idx_c = np.arange(0, n, int(FS / 5))
    idx_a = np.arange(0, n, int(FS))
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
    duree = (t_all[-1] - t_all[0]) / 60
    print(f"✓ {sortie}  ({sortie.stat().st_size / 1e6:.1f} Mo)")
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
    a = ap.parse_args()
    c = COMMUNES[a.commune]
    sortie = Path(a.sortie) if a.sortie else RACINE / "donnees_test" / f"{c['nom']}.tdms"
    generer(sortie, a.passages, a.graine, c["A"], c["B"])


if __name__ == "__main__":
    main()
