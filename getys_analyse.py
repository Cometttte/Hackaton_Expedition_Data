# -*- coding: utf-8 -*-
"""
GETYS - analyse automatique d'une tournée de relevé luxmétrique.

Entrées : un fichier .tdms (acquisition NI : luxmètres 50 Hz, GNSS, IMU, chromamètre)
          et l'export Excel des ouvrages (onglet « SUPPORT ECLAIRAGE PUBLIC »).
Sortie  : un dictionnaire de résultats (sérialisable JSON) + exports CSV / GeoJSON.

Toutes les étapes sont vectorisées (numpy / scipy) : quelques dizaines de secondes
pour une tournée de 3 à 4 heures.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from pyproj import Transformer
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths
from scipy.spatial import cKDTree

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Paris")
except Exception:  # pragma: no cover
    TZ = timezone(timedelta(hours=1))

TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
TO_WGS = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)


# --------------------------------------------------------------------------- paramètres
@dataclass
class Params:
    pas_m: float = 2.0             # ré-échantillonnage spatial : 1 point tous les 2 m parcourus
    vitesse_min: float = 0.5       # m/s, en dessous : véhicule à l'arrêt, mesures exclues
    maille_m: float = 10.0         # maille carrée interne pour comparer les passages (non affichée)
    gap_passage_s: float = 20.0    # deux points d'une même maille séparés de plus de 20 s = 2 passages
    seuil_emoy_lx: float = 10.0    # repère d'éclairement (hypothèse en attendant les classes de voie)
    heure_abaissement: float = 22.0   # heure locale du passage à 50 % des luminaires en variation
    # --- lampadaires estimés à partir de l'éclairement
    prominence_min_lx: float = 3.0    # un pic doit dépasser son voisinage d'au moins 3 lx (et 30 % de sa hauteur)
    rayon_regroupement_m: float = 6.0  # pics de passages différents à moins de 6 m sur la trace = même point lumineux
    seuil_asym_cote: float = 0.04     # asymétrie gauche/droite moyenne en dessous : côté non tranché
    accord_cote_min: float = 0.7      # part des passages qui doivent voir le feu du même côté
    dist_laterale_m: float = 5.3      # distance typique trace -> mât (non mesurable aux luxmètres)
    troncon_m: float = 100.0          # longueur des tronçons pour qualifier l'implantation (un côté / deux côtés)
    # --- calage longitudinal capteurs / antenne GPS (remorque + latence d'horodatage)
    calage_auto: bool = True          # estime automatiquement le décalage et la latence sur les allers-retours
    decalage_long_m: float = 0.0      # si calage_auto = False : capteurs en arrière (+) ou en avant (-) de l'antenne
    latence_s: float = 0.0            # si calage_auto = False : retard des horodatages lux sur le GPS
    recalage_est_m: float = 0.0       # décalage d'affichage sur l'image satellite (réglable dans carte.html)
    recalage_nord_m: float = 0.0
    # --- filtrage des pics aberrants
    lux_max_plausible: float = 150.0  # au-dessus : valeur physiquement non plausible pour de l'EP
    hauteur_capteurs_m: float = 1.5   # luxmètres sur la remorque, à 1,5 m du sol
    hauteur_feu_min_m: float = 3.0    # feu le plus bas considéré en voirie (hors bornes basses)
    pic_largeur_min_m: float = -1.0   # largeur mini d'un vrai pic ; < 0 = déduite de la géométrie (voir clean_lux)
    pic_duree_min_s: float = 0.1      # à très basse vitesse : durée minimale à mi-hauteur d'un vrai pic
    ecart_capteurs_max: float = 1.0   # (max - min) / médiane des 3 capteurs au-delà : capteurs incohérents
    ratio_passage_max: float = 3.0    # passage d'une maille > 3 x la médiane des autres : incohérent


# --------------------------------------------------------------------------- outils
def _t():
    return time.perf_counter()


def _epoch(dt64) -> np.ndarray:
    """datetime64 -> secondes epoch (float)."""
    return np.asarray(dt64, dtype="datetime64[us]").astype("int64") / 1e6


def _nmea_deg(v: str, h: str) -> float:
    if not v:
        return math.nan
    d = int(float(v) / 100)
    m = float(v) - 100 * d
    x = d + m / 60
    return -x if h in ("S", "W") else x


def _hms(s: str) -> float:
    return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + float(s[4:])


def _q(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else None


def _r(x, d=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), d)


# --------------------------------------------------------------------------- GNSS
def parse_gga_stream(lines, with_prdid=False):
    """Retourne un DataFrame (t_utc_s, lat, lon, q, nsat, hdop, heading_imu, pitch, roll)."""
    rows = []
    hdg = pitch = roll = math.nan
    day = None
    for l in lines:
        if with_prdid and l.startswith("$PRDID"):
            p = l.split("*")[0].split(",")
            try:
                pitch, roll, hdg = float(p[1]), float(p[2]), float(p[3])
            except (ValueError, IndexError):
                pass
        elif l.startswith("$GPGGA") or l.startswith("$GNGGA"):
            p = l.split(",")
            if len(p) < 9 or not p[2] or p[6] in ("", "0"):
                continue
            try:
                rows.append((_hms(p[1]), _nmea_deg(p[2], p[3]), _nmea_deg(p[4], p[5]),
                             int(p[6]), int(p[7] or 0), float(p[8] or "nan"), hdg, pitch, roll))
            except ValueError:
                continue
        elif day is None and (l.startswith("$GPZDA") or l.startswith("$GNZDA")):
            p = l.split(",")
            try:
                day = datetime(int(p[4]), int(p[3]), int(p[2]), tzinfo=timezone.utc)
            except (ValueError, IndexError):
                pass
    df = pd.DataFrame(rows, columns=["tod", "lat", "lon", "q", "nsat", "hdop", "heading_imu", "pitch", "roll"])
    return df, day


def load_gnss(tdms: TdmsFile, log):
    """GNSS 5 Hz (port C, avec cap IMU) si disponible, sinon GNSS 1 Hz (port A).
    Calcule aussi le décalage d'horloge PC - UTC GPS à partir des horodatages de réception."""
    groups = {g.name: g for g in tdms.groups()}
    # --- port A : horodatage PC message par message -> décalage d'horloge
    gA = groups.get("GNSS")
    dfA, dayA, offset = None, None, None
    if gA is not None and "Port A" in gA:
        msgs = gA["Port A"][:]
        pc = _epoch(gA["date Port A"][:])
        dfA, dayA = parse_gga_stream([str(m) for m in msgs])
        # décalage : pour chaque GGA, horodatage PC de réception - heure UTC du message
        tods, pcs = [], []
        for m, tpc in zip(msgs, pc):
            m = str(m)
            if m.startswith("$GPGGA"):
                p = m.split(",")
                if p[1]:
                    tods.append(_hms(p[1]))
                    pcs.append(tpc)
        if dayA is not None and tods:
            day0 = dayA.timestamp()
            tods = np.array(tods)
            pcs = np.array(pcs)
            gps = day0 + tods
            gps[gps < pcs - 43200] += 86400
            offset = float(np.median(pcs - gps))
    # --- port C : GNSS 5 Hz + PRDID (tangage, roulis, cap)
    dfC, dayC = None, None
    gC = groups.get("IMU")
    if gC is not None and "Port C" in gC:
        raw = bytes(gC["Port C"][:]).decode("ascii", "replace").split("\r\n")
        dfC, dayC = parse_gga_stream(raw, with_prdid=True)
    use_c = dfC is not None and len(dfC) > (len(dfA) if dfA is not None else 0)
    df = dfC if use_c else dfA
    day = (dayC if use_c else dayA) or dayA or dayC
    src = "port C (5 Hz, avec centrale inertielle)" if use_c else "port A (1 Hz)"
    if df is None or day is None or df.empty:
        raise RuntimeError("Aucune position GNSS exploitable dans le fichier TDMS")
    df = df.drop_duplicates("tod", keep="first").copy()
    t = day.timestamp() + df["tod"].to_numpy()
    # passage de minuit
    t[np.r_[False, np.diff(t) < -43200].cumsum() > 0] += 86400
    df["t"] = t
    df = df.sort_values("t").reset_index(drop=True)
    x, y = TO_L93.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    # vitesse et cap à partir des positions (lissées sur ~1 s)
    k = 5 if use_c else 1
    xs = uniform_filter1d(x, k) if k > 1 else x
    ys = uniform_filter1d(y, k) if k > 1 else y
    dt = np.gradient(df["t"].to_numpy())
    dt[dt <= 0] = np.nan
    vx, vy = np.gradient(xs) / dt, np.gradient(ys) / dt
    df["v"] = np.nan_to_num(np.hypot(vx, vy))
    df["cap"] = np.degrees(np.arctan2(vx, vy)) % 360  # 0 = nord, sens horaire
    if offset is None:
        offset = 0.0
        log("⚠ décalage d'horloge PC/GPS non calculable, supposé nul")
    log(f"GNSS : {len(df)} positions via {src}, décalage horloge PC - GPS = {offset:+.1f} s")
    return df, offset, src, day


# --------------------------------------------------------------------------- luxmètres
def load_lux(tdms: TdmsFile):
    g = {gg.name: gg for gg in tdms.groups()}
    grp = None
    for name, gg in g.items():
        if name.lower().startswith("lux"):
            grp = gg
    if grp is None:
        raise RuntimeError("Groupe « Luxmètres » introuvable dans le TDMS")
    chans = grp.channels()[:3]
    meta = []
    arrs = []
    for c in chans:
        p = c.properties
        arrs.append(np.asarray(c[:], float))
        pos = str(p.get("lux_position", "[0;0]")).strip("[]").replace(",", ".").split(";")
        try:
            pos = [float(pos[0]), float(pos[1])]
        except (ValueError, IndexError):
            pos = [0.0, 0.0]
        meta.append(dict(nom=c.name, offset=_r(p.get("lux_offset", 0.0), 2), position=pos))
    n = min(len(a) for a in arrs)
    L = np.vstack([a[:n] for a in arrs])
    p0 = chans[0].properties
    t0 = _epoch(np.datetime64(p0["wf_start_time"]))
    dt = float(p0.get("wf_increment", 0.02))
    t_pc = t0 + np.arange(n) * dt
    return L, t_pc, dt, meta


def _runs(mask):
    """Débuts et fins (exclusives) des suites de True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.where(d == 1)[0], np.where(d == -1)[0]


def _interp_masque(v, bad):
    """Remplace les échantillons marqués par une interpolation linéaire entre les voisins valides."""
    if not bad.any() or bad.all():
        return v
    i = np.arange(len(v))
    out = v.copy()
    out[bad] = np.interp(i[bad], i[~bad], v[~bad])
    return out


def clean_lux(L, fs, V=None, P: "Params | None" = None):
    """Nettoyage du signal en 5 étapes (détail dans EXPLICATION_DU_CODE.md) :
    1. négatifs (offset capteur) ramenés à 0 ;
    2. filtre de Hampel par capteur (0,5 s) : parasites isolés sur un seul capteur ;
    3. médiane des 3 capteurs (+ comptage des instants où les capteurs se contredisent) ;
    4. pics aberrants sur le signal combiné, par excursion au-dessus de la ligne de base (2 s) :
       a. valeur > lux_max_plausible            -> « trop haut »
       b. excursion plus étroite que pic_largeur_min_m (ou plus brève que pic_duree_min_s)
          -> « trop bref » : un luminaire éclaire plusieurs mètres de route, pas quelques centimètres ;
    5. les échantillons aberrants sont remplacés par interpolation entre les voisins valides
       (sur le signal combiné et sur chaque capteur)."""
    P = P or Params()
    if P.pic_largeur_min_m < 0:
        # largeur à mi-hauteur sous une source ponctuelle à une hauteur dh au-dessus du capteur : 1,53 x dh
        # (E ∝ dh³/(dh²+x²)^1,5). Feu le plus bas 3 m, capteurs à 1,5 m -> dh = 1,5 m -> 2,3 m ; marge de 20 %.
        larg_min = 0.8 * 1.53 * max(0.5, P.hauteur_feu_min_m - P.hauteur_capteurs_m)
    else:
        larg_min = P.pic_largeur_min_m
    n = L.shape[1]
    dt = 1 / fs
    neg = float((L < 0).mean())
    L = np.clip(L, 0, None)
    # --- 2. Hampel par capteur
    w = max(3, int(round(0.5 * fs)) | 1)
    med = median_filter(L, size=(1, w), mode="nearest")
    dev = np.abs(L - med)
    mad = median_filter(dev, size=(1, w), mode="nearest") * 1.4826
    thr = np.maximum.reduce([np.full_like(med, 8.0), 4 * mad, 0.6 * med])
    spikes = dev > thr
    Lc = np.where(spikes, med, L)
    # --- 3. combinaison + cohérence entre capteurs
    E = np.median(Lc, axis=0)
    spread = Lc.max(0) - Lc.min(0)
    incoh = (spread > 30) & (spread > P.ecart_capteurs_max * np.maximum(E, 1))
    # --- 4. pics aberrants : chaque pic du signal combiné est mesuré (hauteur, proéminence,
    #        largeur à mi-hauteur convertie en mètres avec la vitesse du véhicule)
    v = np.full(n, 10.0) if V is None else np.nan_to_num(V, nan=0.0)
    pk, pr = find_peaks(E, prominence=15)
    prom = pr["prominences"]
    base_pk = E[pk] - prom
    # candidats : pics qui au moins doublent le niveau ambiant, ou trop hauts
    cand = (E[pk] > 2 * np.maximum(base_pk, 1)) | (E[pk] > P.lux_max_plausible)
    pk, prom, base_pk = pk[cand], prom[cand], base_pk[cand]
    bad = np.zeros(n, bool)
    events = []
    n_haut = n_bref = n_arret = 0
    if len(pk):
        w50, _, l50, r50 = peak_widths(E, pk, rel_height=0.5)
        _, _, l95, r95 = peak_widths(E, pk, rel_height=0.95)
        dur = w50 * dt
        larg = dur * v[pk]
        trop_haut = E[pk] > P.lux_max_plausible
        trop_bref = ~trop_haut & ((larg < larg_min) | (dur < P.pic_duree_min_s))
        for k in np.where(trop_haut | trop_bref)[0]:
            a_, b_ = int(np.floor(l95[k])), int(np.ceil(r95[k])) + 1
            bad[max(0, a_):min(n, b_)] = True
            lab = ("trop haut" if trop_haut[k] else
                   "variation à l'arrêt" if v[pk[k]] < P.vitesse_min else "trop bref")
            events.append((int(pk[k]), float(E[pk[k]]), float(dur[k]), float(larg[k]), float(base_pk[k]), lab))
        n_haut, n_bref = int(trop_haut.sum()), int(trop_bref.sum())
        n_arret = sum(1 for e in events if e[5] == "variation à l'arrêt")
        n_bref -= n_arret
    E = _interp_masque(E, bad)
    for c in range(Lc.shape[0]):
        Lc[c] = _interp_masque(Lc[c], bad)
    stats = dict(part_negatifs=neg, part_pics=float(spikes.mean()), part_incoherents=float(incoh.mean()),
                 n_trop_haut=n_haut, n_trop_bref=n_bref, n_arret=n_arret,
                 part_aberrants=float(bad.mean()), n_excursions=int(len(pk)), largeur_min_m=larg_min)
    return Lc, E, stats, events, bad


def calage_longitudinal(t, E, V, gnss, log=print):
    """Estime où se trouvent réellement les capteurs le long de la trace GPS.

    Principe : une rue parcourue dans les deux sens voit les mêmes luminaires. Si les mesures sont
    placées trop en avant (ou en arrière) de d mètres, le profil d'éclairement du sens 1 est décalé de +d
    et celui du sens 2 de -d : les deux profils ne se superposent plus. On cherche le décalage qui
    maximise la corrélation entre les deux sens, case par case (1 m), pour 4 classes de vitesse.
    Un décalage qui grandit avec la vitesse est un retard d'horodatage (latence, en s) ; la partie
    constante est un bras de levier fixe (antenne GPS / capteurs de la remorque, en m).
    Retour : (decalage_m, latence_s, détail)."""
    tg = gnss["t"].to_numpy()
    gx, gy = gnss["x"].to_numpy(), gnss["y"].to_numpy()
    stp = np.r_[0, np.hypot(np.diff(gx), np.diff(gy))]
    stp[gnss["v"].to_numpy() < 0.5] = 0
    Sg = np.cumsum(stp) + np.arange(len(stp)) * 1e-6
    capr = np.radians(gnss["cap"].to_numpy())
    k = np.where((t > tg[0]) & (t < tg[-1]) & (V > 2))[0][::5]   # 10 Hz suffit
    tk, Ek, Vk = t[k], E[k], V[k]
    Sk = np.interp(tk, tg, Sg)
    sn, cs = np.interp(tk, tg, np.sin(capr)), np.interp(tk, tg, np.cos(capr))
    axe = np.abs(sn) > np.abs(cs)                       # rue plutôt est-ouest ou nord-sud
    pos = np.where(axe, sn > 0, cs > 0)                 # sens « positif » sur cet axe

    def score(D, m):
        s_ = Sk[m] - D
        x, y = np.interp(s_, Sg, gx), np.interp(s_, Sg, gy)
        key = (np.floor(x).astype(np.int64) * 10_000_000 + np.floor(y).astype(np.int64)) * 4 + axe[m] * 2 + pos[m]
        u, inv = np.unique(key, return_inverse=True)
        sm = np.bincount(inv, Ek[m]) / np.bincount(inv)
        base = u // 2
        # cases présentes dans les deux sens
        o = np.argsort(base, kind="stable")
        b_, sm_, p_ = base[o], sm[o], (u[o] % 2)
        pair = (b_[1:] == b_[:-1])
        a1, a2 = sm_[1:][pair], sm_[:-1][pair]
        if len(a1) < 200:
            return np.nan, 0
        return float(np.corrcoef(a1, a2)[0, 1]), int(len(a1))

    Ds = np.arange(-8, 8.01, 0.25)
    res = []
    for lo, hi in [(2, 5), (5, 8), (8, 11), (11, 20)]:
        m = (Vk >= lo) & (Vk < hi)
        if m.sum() < 2000:
            continue
        sc = np.array([score(D, m)[0] for D in Ds])
        if np.all(np.isnan(sc)):
            continue
        i = int(np.nanargmax(sc))
        d_ = Ds[i]
        if 0 < i < len(Ds) - 1 and np.isfinite(sc[i - 1]) and np.isfinite(sc[i + 1]):   # sommet parabolique
            den = sc[i - 1] - 2 * sc[i] + sc[i + 1]
            if den < 0:
                d_ = Ds[i] + 0.25 * 0.5 * (sc[i - 1] - sc[i + 1]) / den
        sc0 = sc[np.argmin(np.abs(Ds))]
        res.append((float(np.median(Vk[m])), float(d_), float(sc[i]), float(sc0), int(m.sum())))
    if len(res) < 2:
        log("⚠ calage longitudinal : pas assez d'allers-retours, aucun décalage appliqué")
        return 0.0, 0.0, dict(classes=res, ok=False)
    v_, d_, w_ = np.array([r[0] for r in res]), np.array([r[1] for r in res]), np.array([r[4] for r in res])
    b, a = np.polyfit(v_, d_, 1, w=np.sqrt(w_))         # décalage(v) = a + b·v
    # décalage(v) = a + b·v  <=>  capteur réellement à S(t + latence) - a, avec latence = -b
    log(f"Calage capteurs : bras de levier {a:+.2f} m, latence d'horodatage {-b:+.3f} s")
    return float(a), float(-b), dict(classes=[[_r(r[0], 1), _r(r[1], 2), _r(r[2], 3), _r(r[3], 3), r[4]] for r in res],
                                    ok=True)


# --------------------------------------------------------------------------- inventaire
# --------------------------------------------------------------------------- pipeline
def analyser(tdms_path: Path, out_dir: Path, P: Params | None = None, log=print):
    P = P or Params()
    T = {}
    t0 = _t()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- lecture
    tdms = TdmsFile.read(tdms_path)
    T["lecture_tdms"] = _t() - t0
    t1 = _t()
    gnss, offset, gnss_src, day = load_gnss(tdms, log)
    L, t_pc, dt, lux_meta = load_lux(tdms)
    fs = 1 / dt
    T["gnss"] = _t() - t1

    t_gps = t_pc - offset

    # ---------------- spatialisation (interpolation des positions sur les instants lux)
    t1 = _t()
    tg = gnss["t"].to_numpy()
    ok = (t_gps >= tg[0]) & (t_gps <= tg[-1])
    # trous GNSS > 2 s : on n'interpole pas au travers
    gaps = np.diff(tg) > 2.0
    idx = np.clip(np.searchsorted(tg, t_gps) - 1, 0, len(tg) - 2)
    ok &= ~gaps[idx]
    X = np.interp(t_gps, tg, gnss["x"].to_numpy())
    Y = np.interp(t_gps, tg, gnss["y"].to_numpy())
    V = np.interp(t_gps, tg, gnss["v"].to_numpy())
    capr = np.radians(gnss["cap"].to_numpy())
    CAP = np.degrees(np.arctan2(np.interp(t_gps, tg, np.sin(capr)), np.interp(t_gps, tg, np.cos(capr)))) % 360
    T["interpolation"] = _t() - t1

    # ---------------- nettoyage du signal (a besoin de la vitesse pour juger la largeur des pics)
    t1 = _t()
    Lc, E, bruit, ev_aberrants, bad = clean_lux(L, fs, V, P)
    T["filtrage"] = _t() - t1

    # ---------------- calage capteurs / GPS (remorque, latence) puis repositionnement
    t1 = _t()
    if P.calage_auto:
        dec_m, lat_s, calage = calage_longitudinal(t_gps, E, V, gnss, log)
    else:
        dec_m, lat_s, calage = P.decalage_long_m, P.latence_s, dict(ok=None, classes=[])
    calage.update(decalage_m=_r(dec_m, 2), latence_s=_r(lat_s, 3))
    # position du capteur = point de la trace de l'antenne situé à (distance parcourue à t + latence) - décalage.
    # On suit la trace réelle : dans un virage, la remorque passe là où la voiture est passée.
    t_gps = t_gps + lat_s
    gx_, gy_ = gnss["x"].to_numpy(), gnss["y"].to_numpy()
    stp_ = np.r_[0, np.hypot(np.diff(gx_), np.diff(gy_))]
    stp_[gnss["v"].to_numpy() < 0.5] = 0
    Sg_ = np.cumsum(stp_) + np.arange(len(stp_)) * 1e-6
    s_cap = np.interp(t_gps, tg, Sg_) - dec_m
    X = np.interp(s_cap, Sg_, gx_)
    Y = np.interp(s_cap, Sg_, gy_)
    V = np.interp(t_gps, tg, gnss["v"].to_numpy())
    CAP = np.degrees(np.arctan2(np.interp(t_gps, tg, np.sin(capr)), np.interp(t_gps, tg, np.cos(capr)))) % 360
    T["calage"] = _t() - t1
    t1 = _t()
    moving = ok & (V >= P.vitesse_min)
    part_arret = float(1 - moving[ok].mean()) if ok.any() else 0.0
    # distance cumulée (uniquement en mouvement)
    step = np.r_[0, np.hypot(np.diff(X), np.diff(Y))]
    step[~moving] = 0
    step[step > 5] = 0  # sauts après un trou de données
    S = np.cumsum(step)
    dist_km = float(S[-1] / 1000)
    # ré-échantillonnage spatial : un point tous les pas_m mètres (moyenne des échantillons du pas)
    b = np.floor(S / P.pas_m).astype(np.int64)
    bm = b[moving]
    nb = int(bm.max()) + 1 if bm.size else 0
    cnt = np.bincount(bm, minlength=nb)
    keep = cnt > 0

    def bmean(v):
        return (np.bincount(bm, weights=v[moving], minlength=nb)[keep] / cnt[keep])
    pts = pd.DataFrame(dict(x=bmean(X), y=bmean(Y), E=bmean(E), t=bmean(t_gps), v=bmean(V),
                            L1=bmean(Lc[0]), L2=bmean(Lc[1]), L3=bmean(Lc[2])))
    # cap moyen (circulaire)
    sn = np.bincount(bm, weights=np.sin(np.radians(CAP[moving])), minlength=nb)[keep]
    cs = np.bincount(bm, weights=np.cos(np.radians(CAP[moving])), minlength=nb)[keep]
    pts["cap"] = np.degrees(np.arctan2(sn, cs)) % 360
    # max brut (avant filtrage) dans le pas, pour info
    pts["Emax_brut"] = 0.0
    Eraw = np.median(np.clip(L, 0, None), axis=0)
    mx = np.zeros(nb)
    np.maximum.at(mx, bm, Eraw[moving])
    pts["Emax_brut"] = mx[keep]
    loc = pd.to_datetime(pts["t"], unit="s", utc=True).dt.tz_convert(TZ)
    pts["h_loc"] = loc.dt.hour + loc.dt.minute / 60 + loc.dt.second / 3600
    # passages (runs continus) : coupure si > 10 s sans point
    pts["run"] = np.r_[0, np.cumsum(np.diff(pts["t"].to_numpy()) > 10)]
    T["spatialisation"] = _t() - t1
    log(f"Tournée : {dist_km:.1f} km, {len(pts)} points tous les {P.pas_m:g} m, arrêts {part_arret:.0%}")

    # ---------------- qualité des données
    t1 = _t()
    sub = slice(None, None, 25)
    corr = np.corrcoef(Lc[:, moving][:, ::25])
    cal = None
    groups = {g.name: g for g in tdms.groups()}
    if "Calorimetre" in groups:
        gcal = groups["Calorimetre"]
        tcal = _epoch(gcal["Date"][:]) - offset
        n_ = len(tcal)
        duv = np.asarray(gcal["deltaUV"][:n_], float) if "deltaUV" in [c.name for c in gcal.channels()] else np.zeros(n_)
        cal = pd.DataFrame(dict(t=tcal, Ev=np.asarray(gcal["Ev"][:n_], float), Tcp=np.asarray(gcal["Tcp"][:n_], float), duv=duv))
    corr_cal = None
    if cal is not None and len(cal) > 10:
        Ei = np.interp(cal["t"], t_gps, E)
        m = (cal["t"] > t_gps[0]) & (cal["t"] < t_gps[-1])
        if m.sum() > 10:
            corr_cal = float(np.corrcoef(cal["Ev"][m], Ei[m])[0, 1])
    hloc0 = datetime.fromtimestamp(float(t_gps[ok][0]), TZ)
    hloc1 = datetime.fromtimestamp(float(t_gps[ok][-1]), TZ)
    def hloc_(tt_):
        d_ = datetime.fromtimestamp(float(tt_), TZ)
        return d_.strftime("%Hh%M:%S")
    qual = dict(
        gnss_source=gnss_src, n_pos=int(len(gnss)),
        q_counts={str(k): int(v) for k, v in gnss["q"].value_counts().items()},
        hdop_med=_r(gnss["hdop"].median(), 2), nsat_med=_r(gnss["nsat"].median(), 0),
        decalage_horloge_s=_r(offset, 1), part_negatifs=_r(100 * bruit["part_negatifs"], 2),
        part_pics=_r(100 * bruit["part_pics"], 2), part_incoherents=_r(100 * bruit["part_incoherents"], 2),
        n_trop_haut=bruit["n_trop_haut"], n_trop_bref=bruit["n_trop_bref"], n_var_arret=bruit["n_arret"],
        part_aberrants=_r(100 * bruit["part_aberrants"], 3), n_excursions=bruit["n_excursions"],
        aberrants=[[hloc_(t_gps[i0]), _r(pk_, 0), _r(du_, 2), _r(la_, 1), _r(ba_, 0), ty_]
                   for i0, pk_, du_, la_, ba_, ty_ in sorted(ev_aberrants, key=lambda e: -e[1])[:25]],
        part_arret=_r(100 * part_arret, 1),
        corr_capteurs=[_r(corr[0, 1], 2), _r(corr[0, 2], 2), _r(corr[1, 2], 2)],
        corr_chromametre=_r(corr_cal, 2), capteurs=lux_meta, fs=_r(fs, 1),
        trous_gnss=int(gaps.sum()), duree_h=_r((t_gps[ok][-1] - t_gps[ok][0]) / 3600, 2),
    )

    # méthode statistique la plus représentative : pondération temps vs distance vs passage
    Em = E[moving]
    stats_methodes = dict(
        temps_brut=dict(moy=_r(np.clip(Eraw[ok], 0, None).mean()), med=_r(np.median(Eraw[ok]))),
        temps_filtre=dict(moy=_r(Em.mean()), med=_r(np.median(Em))),
        distance=dict(moy=_r(pts["E"].mean()), med=_r(pts["E"].median())),
    )
    T["qualite"] = _t() - t1

    # ---------------- sens de passage, sans filaire de voirie : axe principal local (maille de 30 m)
    t1 = _t()
    a2 = np.radians(pts["cap"].to_numpy())
    gk = (np.floor(pts["x"] / 30).astype(np.int64) * 1_000_003 + np.floor(pts["y"] / 30).astype(np.int64)).to_numpy()
    s2 = pd.Series(np.sin(2 * a2)).groupby(gk).transform("mean").to_numpy()
    c2 = pd.Series(np.cos(2 * a2)).groupby(gk).transform("mean").to_numpy()
    axe = 0.5 * np.arctan2(s2, c2)          # axe de la voie (angle depuis le nord, défini à 180° près)
    axe = np.where(np.cos(axe) < 0, axe + np.pi, axe)   # sens 1 = vers le nord (ou l'est si axe est-ouest)
    axe = np.where(np.isclose(np.cos(axe), 0, atol=1e-9) & (np.sin(axe) < 0), axe + np.pi, axe)
    pts["axe"] = axe
    pts["sens"] = np.where(np.cos(a2 - axe) >= 0, 1, 2)

    # ---------------- passages multiples (maille interne de 10 m) : cohérence et répétabilité
    key = (np.floor(pts["x"] / P.maille_m).astype(np.int64) * 1_000_003
           + np.floor(pts["y"] / P.maille_m).astype(np.int64)).to_numpy()
    pts["maille"] = key
    o = np.lexsort((pts["t"].to_numpy(), key))
    ks, ts_ = key[o], pts["t"].to_numpy()[o]
    newp = np.r_[True, (ks[1:] != ks[:-1]) | (np.diff(ts_) > P.gap_passage_s)]
    pid = np.empty(len(o), np.int64)
    pid[o] = np.cumsum(newp) - 1
    pts["passage"] = pid
    pas = pts.groupby("passage").agg(k=("maille", "first"), E=("E", "mean"))
    npas = pas.groupby("k")["E"].transform("size")
    medp = pas.groupby("k")["E"].transform("median")
    # passage incohérent : nettement plus lumineux que les autres passages au même endroit (phares, vitrine...)
    pas["incoherent"] = (npas >= 3) & (pas["E"] > P.ratio_passage_max * medp) & (pas["E"] - medp > 20)
    pts["passage_incoherent"] = pas["incoherent"].reindex(pts["passage"]).to_numpy()
    n_pas_incoh = int(pas["incoherent"].sum())
    pc = pas[~pas["incoherent"]].groupby("k")["E"]
    cvm = (pc.std() / pc.mean())[pc.size() >= 2]
    T["passages"] = _t() - t1

    # ---------------- lampadaires estimés à partir de l'éclairement
    t1 = _t()
    lum, ev, info_cap = estimer_lampadaires(pts[~pts["passage_incoherent"]], cal, lux_meta, P, log)
    T["lampadaires"] = _t() - t1

    # ---------------- abaissement nocturne : mêmes points lumineux vus avant et après l'heure d'abaissement
    ab = lum[lum["prom_avant"].notna() & lum["prom_apres"].notna()] if len(lum) else lum
    ratio = (ab["prom_apres"] / ab["prom_avant"]) if len(ab) else pd.Series(dtype=float)
    abaiss = dict(n=int(len(ab)), ratio_med=_r(ratio.median(), 2) if len(ab) else None,
                  n_abaisses=int((ratio < 0.7).sum()), part_abaisses=_r(100 * (ratio < 0.7).mean(), 0) if len(ab) else None,
                  par_type={t_: [int(len(g_)), _r((g_["prom_apres"] / g_["prom_avant"]).median(), 2)]
                            for t_, g_ in ab.groupby("type")} if len(ab) else {})
    if len(lum):
        r_ = lum["prom_apres"] / lum["prom_avant"]
        lum["abaisse"] = np.where(r_.notna(), r_ < 0.7, None)

    # ---------------- synthèse
    nl = int(lum["n_lum"].sum()) if len(lum) else 0
    cotes = lum.groupby("cote")["n_lum"].sum().to_dict() if len(lum) else {}
    types = []
    for t_, g_ in (lum.groupby("type") if len(lum) else []):
        types.append(dict(type=t_, libelle=TYPES[t_], n=int(g_["n_lum"].sum()), tcp=_r(g_["tcp"].median(), 0),
                          p_med=_r(g_["puissance_w"].median(), 0), p_tot_kw=_r((g_["puissance_w"] * g_["n_lum"]).sum() / 1000, 1),
                          c_tot_kw=_r((g_["conso_w"] * g_["n_lum"]).sum() / 1000, 1), fourchette=list(PUISSANCE_TYPE[t_][1:]),
                          E_pic=_r(g_["E_pic"].median())))
    types.sort(key=lambda d_: -d_["n"])
    p_tot = float((lum["puissance_w"] * lum["n_lum"]).sum() / 1000) if len(lum) else 0.0
    c_tot = float((lum["conso_w"] * lum["n_lum"]).sum() / 1000) if len(lum) else 0.0
    p_lo = float((lum["p_min"] * lum["n_lum"]).sum() / 1000) if len(lum) else 0.0
    p_hi = float((lum["p_max"] * lum["n_lum"]).sum() / 1000) if len(lum) else 0.0
    hist_edges = np.arange(0, 102.5, 2.5)
    hcount, _ = np.histogram(np.clip(pts["E"], 0, 100), bins=hist_edges)
    tsm = pts.assign(m=(pts["h_loc"] * 60).round()).groupby("m")["E"].median()
    tsm = tsm.rolling(5, center=True, min_periods=1).median()

    res = dict(
        meta=dict(
            fichier_tdms=Path(tdms_path).name, ville=Path(tdms_path).stem, date=hloc0.strftime("%d/%m/%Y"),
            date_iso=hloc0.strftime("%Y-%m-%d"), debut=hloc0.strftime("%Hh%M"), fin=hloc1.strftime("%Hh%M"),
            genere=datetime.now(TZ).strftime("%d/%m/%Y %Hh%M"), params=asdict(P),
        ),
        kpi=dict(
            dist_km=_r(dist_km, 1), duree_h=qual["duree_h"], E_med=_r(pts["E"].median()),
            E_moy=_r(pts["E"].mean()), E_p25=_r(pts["E"].quantile(.25)), E_p75=_r(pts["E"].quantile(.75)),
            part_sup_seuil=_r(100 * (pts["E"] >= P.seuil_emoy_lx).mean(), 0), part_sup50=_r(100 * (pts["E"] > 50).mean(), 1),
            part_sombre=_r(100 * (pts["E"] < 1).mean(), 0),
            n_points=int(len(lum)), n_lampadaires=nl, n_pics=int(len(ev)),
            n_gauche=int(cotes.get("gauche", 0)), n_droite=int(cotes.get("droite", 0)),
            n_non_tranche=int(cotes.get("non_tranche", 0)),
            n_troncons=int(lum["troncon"].nunique()) if len(lum) else 0,
            troncons=lum.groupby("troncon")["implantation"].first().value_counts().to_dict() if len(lum) else {},
            n_multi=int((lum["n_passages"] >= 2).sum()) if len(lum) else 0,
            cv_passages_med=_r(100 * lum["cv_passages"].median(), 0) if len(lum) else None,
            cv_maille_med=_r(100 * cvm.median(), 0) if len(cvm) else None,
            interdistance_med=_r(_interdistance(lum), 0),
            puissance_tot_kw=_r(p_tot, 1), puissance_tot_min_kw=_r(p_lo, 0), puissance_tot_max_kw=_r(p_hi, 0),
            conso_tot_kw=_r(c_tot, 1), energie_an_mwh=_r(c_tot * HEURES_AN / 1000, 0),
            n_pas_incoh=n_pas_incoh, n_aberrants=bruit["n_trop_haut"] + bruit["n_trop_bref"] + bruit["n_arret"],
            capteur_gauche=info_cap.get("capteur_gauche"), capteur_droit=info_cap.get("capteur_droit"),
        ),
        qualite=qual, methodes=stats_methodes, calage=calage, abaissement=abaiss, types=types,
        hist=[[float(e), int(c)] for e, c in zip(hist_edges[:-1], hcount)],
        ts=[[_r(m_ / 60, 3), _r(v, 1)] for m_, v in tsm.items()],
        types_libelles=TYPES, puissance_type=PUISSANCE_TYPE, heures_an=HEURES_AN, validation=VALIDATION,
    )
    ox, oy = float(pts["x"].median()), float(pts["y"].median())
    res["map"] = dict(
        origin=[ox, oy],
        P=[[_r(a_ - ox, 1), _r(b_ - oy, 1), _r(e_, 1), _r(h_, 3)]
           for a_, b_, e_, h_ in pts[["x", "y", "E", "h_loc"]].to_numpy()[::2]],
        L=[[_r(l.x - ox, 1), _r(l.y - oy, 1), l.cote, l.type, _r(l.puissance_w, 0), _r(l.E_pic), int(l.n_passages),
            _r(l.tcp, 0), l.implantation]
           for l in lum.itertuples()] if len(lum) else [],
        dl=P.dist_laterale_m,
    )
    res["carte"] = carte_data(pts, lum, info_cap, P)
    t1 = _t()
    exports(out_dir, pts, lum)
    T["exports"] = _t() - t1
    res["temps"] = {k: _r(v, 2) for k, v in T.items()}
    res["temps"]["total"] = _r(_t() - t0, 1)
    (out_dir / "resume.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("map", "carte")},
                                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def _interdistance(lum):
    """Interdistance médiane entre points lumineux voisins (indicateur de densité d'implantation)."""
    if len(lum) < 3:
        return None
    d, _ = cKDTree(lum[["x_trace", "y_trace"]].to_numpy()).query(lum[["x_trace", "y_trace"]].to_numpy(), k=2)
    return float(np.median(d[:, 1]))


def carte_data(pts, lum, info_cap, P):
    """Données compactes pour la carte satellite (WGS84)."""
    lon, lat = TO_WGS.transform(pts["x"].to_numpy(), pts["y"].to_numpy())
    g, d = info_cap.get("capteur_gauche", "L3"), info_cap.get("capteur_droit", "L1")
    Pp = [[round(a, 6), round(b, 6), _r(e, 1), _r(lg, 1), _r(lc, 1), _r(ld, 1), int(sn), _r(cp, 0), _r(h, 3)]
          for a, b, e, lg, lc, ld, sn, cp, h in zip(lon, lat, pts["E"], pts[g], pts["L2"], pts[d], pts["sens"],
                                                   pts["cap"], pts["h_loc"])]
    Lm = []
    if len(lum):
        ln, lt = TO_WGS.transform(lum["x"].to_numpy(), lum["y"].to_numpy())
        for k, l in enumerate(lum.itertuples()):
            Lm.append([[round(lt[k], 6), round(ln[k], 6)], l.cote, l.type, _r(l.puissance_w, 0), _r(l.p_min, 0), _r(l.p_max, 0), _r(l.E_pic),
                       int(l.n_passages), _r(l.tcp, 0), _r(l.part_tcp0, 2), _r(l.accord_cote, 2),
                       None if l.abaisse is None else int(bool(l.abaisse)), l.implantation, _r(l.conso_w, 0)])
    return dict(P=Pp, L=Lm, types=TYPES, capteurs=["gauche (" + g.replace("L", "Lux") + ")", "centre (Lux2)",
                                                   "droite (" + d.replace("L", "Lux") + ")"])


def exports(out_dir: Path, pts, lum):
    lon, lat = TO_WGS.transform(pts["x"].to_numpy(), pts["y"].to_numpy())
    e = pts.assign(lon=lon, lat=lat)
    e["heure_locale"] = e["h_loc"].round(4)
    e[["lon", "lat", "x", "y", "E", "L1", "L2", "L3", "Emax_brut", "v", "cap", "sens", "heure_locale", "passage"]] \
        .round(dict(lon=7, lat=7, x=2, y=2, E=2, L1=2, L2=2, L3=2, Emax_brut=2, v=2, cap=1)) \
        .to_csv(out_dir / "mesures_pas2m.csv", index=False, sep=";", decimal=",")
    if not len(lum):
        return
    ln, lt = TO_WGS.transform(lum["x"].to_numpy(), lum["y"].to_numpy())
    l2 = lum.assign(lon=np.round(ln, 7), lat=np.round(lt, 7), type_libelle=lum["type"].map(TYPES))
    cols = ["lon", "lat", "x", "y", "x_trace", "y_trace", "cote", "implantation", "n_lum", "n_passages", "n_pics", "E_pic", "prom", "fwhm", "asym",
            "accord_cote", "type", "type_libelle", "tcp", "part_tcp0", "duv", "puissance_w", "p_min", "p_max", "conso_w",
            "prom_avant", "prom_apres", "abaisse", "cv_passages", "h_min", "h_max"]
    l2[cols].round(dict(x=2, y=2, x_trace=2, y_trace=2, conso_w=0, E_pic=1, prom=1, fwhm=1, asym=3, accord_cote=2, tcp=0, part_tcp0=2, duv=4,
                        puissance_w=0, p_min=0, p_max=0, prom_avant=1, prom_apres=1, cv_passages=2, h_min=4, h_max=4)) \
        .to_csv(out_dir / "lampadaires_estimes.csv", index_label="id", sep=";", decimal=",")
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[float(a), float(b)]),
                  properties=dict(id=int(i), cote=r.cote, implantation_troncon=r.implantation, type=r.type,
                                  type_libelle=TYPES[r.type], tcp=_r(r.tcp, 0), puissance_w=_r(r.puissance_w, 0),
                                  puissance_min_w=_r(r.p_min, 0), puissance_max_w=_r(r.p_max, 0), E_pic=_r(r.E_pic, 1),
                                  n_passages=int(r.n_passages)))
             for i, a, b, r in zip(l2.index, l2["lon"], l2["lat"], l2.itertuples())]
    (out_dir / "lampadaires_estimes.geojson").write_text(
        json.dumps(dict(type="FeatureCollection", features=feats), ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- lampadaires estimés
# Types de lampe reconnaissables au chromamètre (température de couleur proximale Tcp, écart à la courbe
# des corps noirs deltaUV). Le chromamètre renvoie Tcp = 0 sous ~2 200 K : c'est le sodium haute pression.
TYPES = {
    "sodium": "Sodium haute pression (≈ 2 000 K)",
    "blanc_2700": "Blanc chaud ≈ 2 700 K : LED 2 700 K ou iodures métalliques céramiques",
    "led_3000": "Blanc ≈ 3 000 K : LED",
    "blanc_neutre": "Blanc neutre 3 300 à 3 800 K : vapeur de mercure, iodures ou LED",
    "blanc_froid": "Blanc froid > 3 800 K : LED 4 000 K ou iodures métalliques",
    "inconnu": "Indéterminé (pas de mesure chromamètre exploitable)",
}
# Puissance nominale par type de lampe. L'éclairement au passage ne renseigne presque pas sur la puissance :
# corrélation log(proéminence) / log(puissance) = 0,08 sur Ormesson (hauteur, optique, distance et vieillissement
# pèsent plus que les watts). On attribue donc à chaque lampadaire la puissance moyenne de son type, avec
# la fourchette interquartile observée (étalonnage Ormesson, voir EXPLICATION_DU_CODE.md section 9).
PUISSANCE_TYPE = {"sodium": (108, 100, 150), "blanc_2700": (104, 90, 140), "led_3000": (69, 45, 90),
                  "blanc_neutre": (78, 40, 100), "blanc_froid": (122, 100, 140), "inconnu": (95, 70, 150)}
# consommation réelle / puissance de la lampe (pertes du ballast ferromagnétique des lampes à décharge ;
# l'alimentation des LED est déjà comprise dans leur puissance)
PERTES_BALLAST = {"sodium": 1.12, "blanc_2700": 1.08, "led_3000": 1.0, "blanc_neutre": 1.1, "blanc_froid": 1.05,
                  "inconnu": 1.08}
HEURES_AN = 4100
# Précision constatée en comparant, une seule fois et après coup, les estimations à l'inventaire d'Ormesson
# (tournée du 18/03/2025). L'outil n'utilise pas l'inventaire : ces chiffres sont seulement affichés.
VALIDATION = dict(lignes=[
    ["Nombre de lampadaires", "1 270 estimés pour 1 264 supports à moins de 10 m de la trace dans l'inventaire ; "
                              "72 % des lampadaires estimés tombent sur un support (à 12 m près)"],
    ["Position le long de la voie", "écart médian de 2,8 m avec l'inventaire, dont la position est elle-même imprécise"],
    ["Côté gauche / droite", "juste dans 90 % des cas (737 lampadaires comparés)"],
    ["Un côté ou les deux, par tronçon de 100 m", "juste pour 80 % des tronçons (195 comparés)"],
    ["Sodium ou source blanche", "juste dans 84 % des cas ; 84 % des « LED 3 000 K » sont bien des LED"],
    ["Puissance d'un lampadaire", "écart médian de 20 % (75 % des cas sous 29 %)"],
    ["Puissance totale", "63,6 kW estimés pour 63,5 kW à l'inventaire sur les mêmes supports "
                         "(étalonnage fait sur ces données : à recontrôler dans une autre commune)"],
])   # durée annuelle de fonctionnement usuelle de l'éclairage public en France (h)

def type_lampe(part_tcp0, tcp, duv):
    if part_tcp0 is None or np.isnan(part_tcp0):
        return "inconnu"
    if part_tcp0 >= 0.5:
        return "sodium"
    if tcp is None or np.isnan(tcp):
        return "inconnu"
    if tcp < 2800:
        return "blanc_2700"
    if tcp < 3300:
        return "led_3000"
    if tcp < 3800:
        return "blanc_neutre"
    return "blanc_froid"


def estimer_lampadaires(pts, cal, lux_meta, P, log=print):
    """Lampadaires estimés à partir de l'éclairement seul (aucun inventaire).

    1. Pics : chaque passage sous un lampadaire crée une « cloche » dans l'éclairement filtré.
    2. Côté : le luxmètre gauche et le luxmètre droit sont écartés de 0,98 m ; celui qui est du côté du feu
       reçoit plus de lumière. Asymétrie a = (gauche - droite) / (2 x proéminence du pic).
    3. Regroupement des passages : les pics de tous les passages situés à moins de rayon_regroupement_m sur
       la trace sont un même point lumineux. Le côté est voté sur tous les passages, dans un repère commun.
    4. Position : point de la trace au maximum (précis le long de la voie), décalé de dist_laterale_m vers
       le côté trouvé (distance au bord non mesurable : valeur typique).
    5. Type de lampe : chromamètre autour du maximum. 6. Puissance : valeur moyenne du type (fourchette).
    """
    ys = [m["position"][1] for m in lux_meta]
    ig, id_ = int(np.argmax(ys)), int(np.argmin(ys))      # capteur le plus à gauche / à droite (y > 0 = gauche)
    if ys[ig] == ys[id_]:
        ig, id_ = 2, 0
    cg, cd = f"L{ig + 1}", f"L{id_ + 1}"
    rows = []
    for rid, g in pts.groupby("run"):
        if len(g) < 10:
            continue
        e = g["E"].to_numpy()
        es = uniform_filter1d(e, 3)
        pk, pr = find_peaks(es, prominence=P.prominence_min_lx, distance=4, height=4)
        prom = pr["prominences"]
        sel = prom >= np.maximum(P.prominence_min_lx, 0.3 * es[pk])
        pk, prom = pk[sel], prom[sel]
        if not len(pk):
            continue
        w = peak_widths(es, pk, rel_height=0.5)[0] * P.pas_m
        gi = g.index.to_numpy()
        Lg, Ld = g[cg].to_numpy(), g[cd].to_numpy()
        for j, p in enumerate(pk):
            rows.append((gi[p], rid, es[p], prom[j], w[j], (Lg[p] - Ld[p]) / (2 * prom[j])))
    ev = pd.DataFrame(rows, columns=["i", "run", "E", "prom", "fwhm", "asym"])
    if not len(ev):
        return pd.DataFrame(), ev, dict(n=0)
    ev = ev.join(pts[["x", "y", "cap", "t", "h_loc", "sens", "axe"]], on="i")
    # chromamètre autour du maximum (± 1,5 s, Ev > 3 lx)
    z_, tcp_, duv_ = np.full(len(ev), np.nan), np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    if cal is not None and len(cal):
        tc = cal["t"].to_numpy(); Ev_ = cal["Ev"].to_numpy(); Tc_ = cal["Tcp"].to_numpy(); Du_ = cal["duv"].to_numpy()
        for k, tt in enumerate(ev["t"].to_numpy()):
            a_, b_ = np.searchsorted(tc, tt - 1.5), np.searchsorted(tc, tt + 1.5)
            m_ = Ev_[a_:b_] > 3
            if not m_.any():
                continue
            tw, dw = Tc_[a_:b_][m_], Du_[a_:b_][m_]
            z_[k] = float((tw == 0).mean())
            if (tw > 0).any():
                tcp_[k], duv_[k] = float(np.median(tw[tw > 0])), float(np.median(dw[tw > 0]))
    ev["part_tcp0"], ev["tcp"], ev["duv"] = z_, tcp_, duv_
    # repère commun : direction de l'axe local (sens 1). Gauche du sens 1 = côté +.
    hx, hy = np.sin(np.radians(ev["cap"])), np.cos(np.radians(ev["cap"]))
    ax_x, ax_y = np.sin(ev["axe"]), np.cos(ev["axe"])
    al = np.where(hx * ax_x + hy * ax_y >= 0, 1.0, -1.0)
    ev["asym_s1"] = ev["asym"] * al
    # regroupement glouton par proéminence décroissante
    X, Y = ev["x"].to_numpy(), ev["y"].to_numpy()
    tree = cKDTree(np.c_[X, Y])
    lab = -np.ones(len(ev), int)
    out = []
    for i in np.argsort(-ev["prom"].to_numpy()):
        if lab[i] >= 0:
            continue
        nb = np.array([j for j in tree.query_ball_point([X[i], Y[i]], P.rayon_regroupement_m) if lab[j] < 0])
        lab[nb] = len(out)
        g = ev.iloc[nb]
        s1 = g["asym_s1"].to_numpy()
        m = float(np.mean(s1))
        accord = float(np.mean(np.sign(s1) == np.sign(m))) if m != 0 else 0.5
        n = len(g)
        if abs(m) >= P.seuil_asym_cote and (accord >= P.accord_cote_min or n == 1):
            cote = "gauche" if m > 0 else "droite"
        else:
            # asymétrie faible ou contradictoire : feu au-dessus de l'axe, deux feux face à face, ou feu éloigné.
            # Les trois cas donnent la même mesure (capteurs écartés de 0,98 m seulement) : on ne tranche pas.
            cote = "non_tranche"
        axx, axy = float(ax_x.iloc[nb[0]]), float(ax_y.iloc[nb[0]])
        cx, cy = float(g["x"].mean()), float(g["y"].mean())
        lx, ly = -axy, axx                                  # vecteur gauche du sens 1
        sgn = {"gauche": 1, "droite": -1}.get(cote, 0)
        avant = g[g["h_loc"] < P.heure_abaissement]
        apres = g[g["h_loc"] >= P.heure_abaissement]
        prom_ref = float(np.median(avant["prom"])) if len(avant) else float(np.median(g["prom"]))
        z = float(np.nanmedian(g["part_tcp0"])) if g["part_tcp0"].notna().any() else np.nan
        tcp = float(np.nanmedian(g["tcp"])) if g["tcp"].notna().any() else np.nan
        duv = float(np.nanmedian(g["duv"])) if g["duv"].notna().any() else np.nan
        ty = type_lampe(z, tcp, duv)
        n_lum = 1
        pw, pmin, pmax = PUISSANCE_TYPE[ty]
        cv = float(g["prom"].std() / g["prom"].mean()) if n >= 2 else np.nan
        out.append(dict(x=cx + sgn * P.dist_laterale_m * lx, y=cy + sgn * P.dist_laterale_m * ly,
                        x_trace=cx, y_trace=cy, lx=lx, ly=ly, cote=cote, n_lum=n_lum, n_passages=int(g["run"].nunique()),
                        n_pics=n, asym=m, accord_cote=accord, E_pic=float(g["E"].median()), prom=prom_ref,
                        fwhm=float(g["fwhm"].median()), cv_passages=cv, part_tcp0=z, tcp=tcp, duv=duv, type=ty,
                        puissance_w=float(pw), p_min=float(pmin), p_max=float(pmax),
                        conso_w=float(pw * PERTES_BALLAST[ty]),
                        prom_avant=float(np.median(avant["prom"])) if len(avant) else np.nan,
                        prom_apres=float(np.median(apres["prom"])) if len(apres) else np.nan,
                        h_min=float(g["h_loc"].min()), h_max=float(g["h_loc"].max())))
    lum = pd.DataFrame(out)
    ev["lampadaire"] = lab
    # implantation par tronçon de ~100 m : un seul côté, ou les deux côtés de la voie (bilatéral / quinconce)
    tk = (np.floor(lum["x_trace"] / P.troncon_m).astype(np.int64) * 1_000_003
          + np.floor(lum["y_trace"] / P.troncon_m).astype(np.int64))
    lum["troncon"] = tk
    impl = {}
    for k_, g_ in lum.groupby("troncon"):
        ng, nd = int((g_["cote"] == "gauche").sum()), int((g_["cote"] == "droite").sum())
        if ng + nd < 2:
            impl[k_] = "indetermine"
        elif min(ng, nd) / (ng + nd) >= 0.25:
            impl[k_] = "deux_cotes"
        else:
            impl[k_] = "gauche" if ng > nd else "droite"
    lum["implantation"] = lum["troncon"].map(impl)
    log(f"Lampadaires estimés : {len(ev)} pics -> {len(lum)} points lumineux, {int(lum['n_lum'].sum())} lampadaires")
    return lum, ev, dict(capteur_gauche=cg, capteur_droit=cd)



