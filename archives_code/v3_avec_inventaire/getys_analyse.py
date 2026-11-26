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
    hex_m: float = 10.0            # rayon (centre-sommet) des cellules hexagonales
    gap_passage_s: float = 20.0    # deux points d'une même cellule séparés de plus de 20 s = 2 passages
    rayon_support_m: float = 12.0  # zone d'influence d'un support pour le maximum au passage
    dist_frole_m: float = 10.0     # un support est « frôlé » si la trajectoire passe à moins de 10 m
    seuil_eteint_lx: float = 5.0   # max au passage sous ce seuil : support suspect
    rayon_rue_m: float = 30.0      # rattachement d'une mesure à la rue du support le plus proche
    seuil_emoy_lx: float = 10.0    # cible d'éclairement moyen (hypothèse en attendant le SDAL)
    seuil_cct_sodium: float = 2400.0  # K, en dessous : source sodium probable
    heure_abaissement: float = 22.0   # heure locale du passage à 50 % pour les luminaires en variation
    decalage_lateral_m: float = 3.5   # position estimée d'un luminaire détecté par rapport à l'axe du véhicule
    rayon_match_m: float = 15.0    # appariement pic détecté <-> support inventorié
    # --- recalage des positions de l'inventaire sur les points lumineux mesurés
    recaler_inventaire: bool = True   # refait les mesures par support à la position recalée
    ecart_position_min_m: float = 5.0  # écart inventaire / point lumineux au-delà duquel la position est signalée
    dispersion_pics_max_m: float = 2.0  # écart médian entre passages au-delà duquel le recalage n'est pas fiable
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
    ratio_passage_max: float = 3.0    # passage d'un hexagone > 3 x la médiane des autres : incohérent


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
def recaler_inventaire(sup, ex, ey, hx, hy, side, lab, n_clu, P, log=print):
    """Deux passes : 1) appariement brut, estimation du décalage d'ensemble de l'inventaire ;
    2) si ce décalage dépasse 1 m, tout l'inventaire est translaté puis l'appariement est refait."""
    out, det_sup, resume = _recaler(sup, ex, ey, hx, hy, side, lab, n_clu, P, (0.0, 0.0))
    g = (resume["global_e"] or 0.0, resume["global_n"] or 0.0)
    resume["global_applique"] = bool(resume["global_norme"] and resume["global_norme"] >= 1.0)
    if resume["global_applique"]:
        out, det_sup, r2 = _recaler(sup, ex, ey, hx, hy, side, lab, n_clu, P, g)
        r2.update(global_e=resume["global_e"], global_n=resume["global_n"], global_norme=resume["global_norme"],
                  global_applique=True, global_residuel=r2["global_norme"])
        resume = r2
    log(f"Recalage inventaire : décalage d'ensemble {resume['global_norme']} m "
        f"({'appliqué' if resume['global_applique'] else 'non appliqué'}), "
        f"{resume['n_apparies']} supports appariés à un point lumineux ({resume['n_fiables']} fiables), "
        f"écart médian {resume['ecart_med']} m, {resume['n_douteux']} positions à corriger "
        f"(≥ {P.ecart_position_min_m:g} m)")
    return out, det_sup, resume


def _recaler(sup, ex, ey, hx, hy, side, lab, n_clu, P, glob):
    """Recale chaque support de l'inventaire sur le point lumineux réellement mesuré.

    Entrées : un « événement » par pic de lumière détecté à un passage (position du véhicule ex, ey au pic,
    direction de marche hx, hy, côté du luminaire side = +1 gauche / -1 droite, groupe lab = luminaire).
    Principe :
      1. pour chaque couple (support, luminaire détecté) proche, on mesure à chaque passage la distance
         LE LONG DE LA VOIE entre le support et le pic (c'est la composante précise : le pic se produit quand
         la remorque passe au droit du feu ; la distance latérale, elle, n'est pas mesurable avec les luxmètres) ;
      2. affectation optimale un-pour-un (algorithme hongrois) : chaque pic va au support qui l'explique
         le mieux (décalage le long de la voie + pénalité si le côté gauche/droite mesuré contredit l'Excel),
         ce qui évite de rattacher un pic au voisin quand la position déclarée est fausse ;
      3. position corrigée = position Excel déplacée le long de la voie jusqu'au pic ; si tous les passages
         voient la lumière du côté opposé, le support est aussi basculé de l'autre côté de la trajectoire.
    """
    from scipy.optimize import linear_sum_assignment
    sx0, sy0 = sup["x_inv"].to_numpy(float), sup["y_inv"].to_numpy(float)
    sx, sy = sx0 + glob[0], sy0 + glob[1]          # inventaire translaté du décalage d'ensemble (passe 2)
    nS = len(sup)
    out = pd.DataFrame(dict(x_corr=sx.copy(), y_corr=sy.copy(), apparie_pic=np.zeros(nS, bool),
                            ecart_position_m=np.full(nS, np.nan), decalage_voie_m=np.full(nS, np.nan),
                            cote_inverse=np.zeros(nS, bool), n_pics=np.zeros(nS, int),
                            dispersion_m=np.full(nS, np.nan), accord_cote=np.full(nS, np.nan),
                            recalage_fiable=np.zeros(nS, bool), position_douteuse=np.zeros(nS, bool)))
    det_sup = -np.ones(n_clu, int)
    resume = dict(n_apparies=0, n_fiables=0, n_douteux=0, n_cote_inverse=0, ecart_med=None, ecart_p90=None,
                  global_e=None, global_n=None, global_norme=None, seuil=P.ecart_position_min_m)
    if n_clu == 0 or len(ex) == 0:
        return out, det_sup, resume
    lat_max = P.dist_frole_m + 5.0
    tree = cKDTree(np.c_[ex, ey])
    cand = tree.query_ball_point(np.c_[sx, sy], r=math.hypot(P.rayon_match_m, lat_max))
    pairs = {}
    for i, ks in enumerate(cand):
        for k in ks:
            pairs.setdefault((i, int(lab[k])), []).append(k)
    info = {}
    for (i, c), ks in pairs.items():
        ks = np.asarray(ks)
        dx_, dy_ = ex[ks] - sx[i], ey[ks] - sy[i]
        along = dx_ * hx[ks] + dy_ * hy[ks]          # pic en avant (+) / en arrière (-) du support, sens de marche
        lat = dx_ * hy[ks] - dy_ * hx[ks]            # support à gauche (+) / droite (-) de la trajectoire
        keep = np.abs(lat) <= lat_max
        if not keep.any():
            continue
        ks, along, lat = ks[keep], along[keep], lat[keep]
        k0 = ks[0]
        al = np.where(hx[ks] * hx[k0] + hy[ks] * hy[k0] >= 0, 1.0, -1.0)   # passages dans l'autre sens
        off, lat_al, side_al = along * al, lat * al, side[ks] * al
        om = float(np.median(off))
        if abs(om) > P.rayon_match_m:
            continue
        accord = float(np.mean(side_al == np.sign(lat_al)))
        info[(i, c)] = dict(k0=k0, off=om, mad=float(np.median(np.abs(off - om))), n=len(ks),
                            lat=float(np.median(lat_al)), accord=accord, cout=abs(om) + 4.0 * (1 - accord))
    if not info:
        return out, det_sup, resume
    ri = sorted({i for i, _ in info}); ci = sorted({c for _, c in info})
    rpos = {i: n for n, i in enumerate(ri)}; cpos = {c: n for n, c in enumerate(ci)}
    M = np.full((len(ri), len(ci)), 1e6)
    for (i, c), d_ in info.items():
        M[rpos[i], cpos[c]] = d_["cout"]
    rr, cc = linear_sum_assignment(M)
    A, b_ = [], []
    for r_, c_ in zip(rr, cc):
        if M[r_, c_] >= 1e5:
            continue
        i, c = ri[r_], ci[c_]
        d_ = info[(i, c)]
        k0 = d_["k0"]
        # bascule de côté seulement si tous (≥ 75 %) les passages voient la lumière de l'autre côté et que le
        # support est en bord de voie (≤ 8 m) : au-delà, la lumière vue peut venir d'une autre source
        inv = d_["n"] >= 2 and d_["accord"] <= 0.25 and abs(d_["lat"]) <= 8.0
        mx = d_["off"] * hx[k0] + (-2 * d_["lat"] * (-hy[k0]) if inv else 0.0)
        my = d_["off"] * hy[k0] + (-2 * d_["lat"] * hx[k0] if inv else 0.0)
        fiable = d_["n"] >= 2 and d_["mad"] <= P.dispersion_pics_max_m
        e_ = math.hypot(mx + glob[0], my + glob[1])     # écart total à la position de l'Excel
        out.loc[i, ["x_corr", "y_corr", "ecart_position_m", "decalage_voie_m", "dispersion_m", "accord_cote"]] = \
            [sx[i] + mx, sy[i] + my, e_, d_["off"], d_["mad"], d_["accord"]]
        out.loc[i, ["apparie_pic", "cote_inverse", "recalage_fiable", "position_douteuse"]] = \
            [True, inv, fiable, fiable and e_ >= P.ecart_position_min_m]
        out.loc[i, "n_pics"] = d_["n"]
        det_sup[c] = i
        if fiable:
            A.append([hx[k0], hy[k0]]); b_.append(d_["off"])
    # décalage d'ensemble de l'inventaire (moindres carrés robustes sur les décalages le long des voies :
    # les rues ayant des orientations variées, la translation Est/Nord est déterminée)
    if len(A) >= 20:
        A, b_ = np.asarray(A), np.asarray(b_)
        if np.linalg.eigvalsh(A.T @ A / len(A)).min() > 0.1:
            w = np.ones(len(b_))
            for _ in range(10):
                sv = np.linalg.lstsq(A * w[:, None], b_ * w, rcond=None)[0]
                res_ = np.abs(b_ - A @ sv)
                w = np.sqrt(np.minimum(1.0, 2.0 / np.maximum(res_, 1e-9)))   # Huber, seuil 2 m
            resume.update(global_e=_r(sv[0], 2), global_n=_r(sv[1], 2), global_norme=_r(math.hypot(*sv), 2))
    f_ = out[out["recalage_fiable"]]
    resume.update(n_apparies=int(out["apparie_pic"].sum()), n_fiables=int(len(f_)),
                  n_douteux=int(out["position_douteuse"].sum()), n_cote_inverse=int(out["cote_inverse"].sum()),
                  ecart_med=_r(f_["ecart_position_m"].median(), 1) if len(f_) else None,
                  ecart_p90=_r(f_["ecart_position_m"].quantile(0.9), 1) if len(f_) else None)
    return out, det_sup, resume


def load_inventaire(xlsx: Path, date_releve: datetime):
    xl = pd.ExcelFile(xlsx)
    sheet = next((s for s in xl.sheet_names if "SUPPORT ECLAIRAGE PUBLIC" in s.upper()), None)
    if sheet is None:
        raise RuntimeError("Onglet « SUPPORT ECLAIRAGE PUBLIC » introuvable dans l'Excel")
    d = xl.parse(sheet, header=1)
    d = d[d["LATITUDE"].notna() & d["LONGITUDE"].notna()].copy()
    for c in ["PUISSANCE NOMINALE SOURCE W", "HAUTEUR SUPPORT M", "TEMPERATURE DE COULEUR K",
              "FLUX LUMINEUX LM", "AVANCEE CONSOLE M", "HAUTEUR SOURCE M"]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    x, y = TO_L93.transform(d["LONGITUDE"].to_numpy(float), d["LATITUDE"].to_numpy(float))
    d["x"], d["y"] = x, y
    dref = pd.Timestamp(date_releve.date())
    dp = pd.to_datetime(d.get("DATE POSE LUMINAIRE"), errors="coerce")
    ds = pd.to_datetime(d.get("DATE POSE SOURCE"), errors="coerce")
    d["modif_apres_releve"] = (dp > dref) | (ds > dref)
    d["annee_pose"] = dp.dt.year
    d["FAMILLE SOURCE"] = d["FAMILLE SOURCE"].fillna("NON RENSEIGNEE")
    d["fam3"] = np.select([d["FAMILLE SOURCE"].str.contains("LED"),
                           d["FAMILLE SOURCE"].str.contains("SODIUM")], [0, 1], 2)
    # une ligne par luminaire -> une ligne par support (ID)
    agg = dict(NUMERO="first", ZONE="first", PARENT="first", x="first", y="first",
               LATITUDE="first", LONGITUDE="first")
    sup = d.groupby("ID", sort=False).agg(agg)
    first = d.groupby("ID", sort=False).first()
    sup["famille"] = first["FAMILLE SOURCE"]
    sup["fam3"] = first["fam3"]
    sup["puissance_w"] = d.groupby("ID", sort=False)["PUISSANCE NOMINALE SOURCE W"].sum(min_count=1)
    sup["nb_lum"] = d.groupby("ID", sort=False).size()
    sup["etat"] = first["ETAT LUMINAIRE"]
    sup["regime"] = first.get("REGIME D ALLUMAGE")
    sup["categorie"] = first.get("CATEGORIE OUVRAGE")
    sup["hauteur"] = first.get("HAUTEUR SUPPORT M")
    sup["type_lum"] = first.get("TYPE LUMINAIRE")
    sup["k_decl"] = first.get("TEMPERATURE DE COULEUR K")
    sup["modif_apres_releve"] = d.groupby("ID", sort=False)["modif_apres_releve"].any()
    sup = sup.reset_index()
    return d, sup, sheet


def synthese_inventaire(d: pd.DataFrame):
    def vc(col, n=None):
        if col not in d:
            return {}
        s = d[col].fillna("Non renseigné").astype(str).value_counts()
        return {k: int(v) for k, v in (s.head(n) if n else s).items()}
    fam = []
    for f, g in d.groupby("FAMILLE SOURCE"):
        fam.append([f, int(len(g)), _r(g["PUISSANCE NOMINALE SOURCE W"].sum() / 1000, 1),
                    _r(g["PUISSANCE NOMINALE SOURCE W"].mean(), 0)])
    fam.sort(key=lambda r: -r[1])
    etat_fam = {f: {k: int(v) for k, v in g["ETAT LUMINAIRE"].value_counts().items()}
                for f, g in d.groupby("FAMILLE SOURCE")}
    annee = {str(int(k)): int(v) for k, v in d["annee_pose"].value_counts().sort_index().items()}
    fill = {c: int(d[c].notna().sum()) for c in ["REGIME D ALLUMAGE", "FLUX LUMINEUX LM",
                                                   "TEMPERATURE DE COULEUR K", "DATE POSE LUMINAIRE",
                                                   "PUISSANCE NOMINALE SOURCE W", "HAUTEUR SUPPORT M"] if c in d}
    return dict(n_lum=int(len(d)), n_sup=int(d["ID"].nunique()), fam=fam, etat_fam=etat_fam,
                etat=vc("ETAT LUMINAIRE"), cat=vc("CATEGORIE OUVRAGE", 8), regime=vc("REGIME D ALLUMAGE"),
                typelum=vc("TYPE LUMINAIRE", 8), annee=annee, annee_na=int(d["annee_pose"].isna().sum()),
                puissance_kw=_r(d["PUISSANCE NOMINALE SOURCE W"].sum() / 1000, 1),
                n_modif_apres=int(d["modif_apres_releve"].sum()), fill=fill,
                n_armoires=int(d["PARENT"].nunique()))


# --------------------------------------------------------------------------- hexagones
def hex_index(x, y, R):
    """Hexagones « pointe en haut », rayon R. Retourne (q, r) axiaux entiers."""
    qf = (math.sqrt(3) / 3 * x - y / 3) / R
    rf = (2 / 3 * y) / R
    sf = -qf - rf
    q, r, s = np.round(qf), np.round(rf), np.round(sf)
    dq, dr, ds = np.abs(q - qf), np.abs(r - rf), np.abs(s - sf)
    m1 = (dq > dr) & (dq > ds)
    m2 = ~m1 & (dr > ds)
    q = np.where(m1, -r - s, q)
    r = np.where(m2, -q - s, r)
    return q.astype(np.int64), r.astype(np.int64)


def hex_center(q, r, R):
    return R * math.sqrt(3) * (q + r / 2), R * 1.5 * r


def hex_polygon(cx, cy, R):
    return [(cx + R * math.cos(math.radians(60 * i + 30)), cy + R * math.sin(math.radians(60 * i + 30)))
            for i in range(6)]


# --------------------------------------------------------------------------- pipeline
def analyser(tdms_path: Path, xlsx_path: Path, out_dir: Path, P: Params | None = None, log=print):
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
        cal = pd.DataFrame(dict(t=tcal, Ev=gcal["Ev"][:], Tcp=gcal["Tcp"][:]))
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

    # ---------------- hexagones et multi-passages
    t1 = _t()
    R = P.hex_m
    q, r = hex_index(pts["x"].to_numpy(), pts["y"].to_numpy(), R)
    pts["hq"], pts["hr"] = q, r
    key = q * 1_000_003 + r
    pts["hk"] = key
    o = np.lexsort((pts["t"].to_numpy(), key))
    ks, ts_ = key[o], pts["t"].to_numpy()[o]
    newp = np.r_[True, (ks[1:] != ks[:-1]) | (np.diff(ts_) > P.gap_passage_s)]
    pid = np.empty(len(o), np.int64)
    pid[o] = np.cumsum(newp) - 1
    pts["passage"] = pid
    pas = pts.groupby("passage").agg(hk=("hk", "first"), hq=("hq", "first"), hr=("hr", "first"),
                                     E=("E", "mean"), Emin=("E", "min"), n=("E", "size"),
                                     t=("t", "mean"), h=("h_loc", "mean"))
    # passage incohérent : dans une cellule vue au moins 3 fois, un passage nettement plus lumineux
    # que les autres (> ratio x médiane et > +20 lx) -> source transitoire (phares, véhicule éclairé...)
    npas = pas.groupby("hk")["E"].transform("size")
    medp = pas.groupby("hk")["E"].transform("median")
    pas["incoherent"] = (npas >= 3) & (pas["E"] > P.ratio_passage_max * medp) & (pas["E"] - medp > 20)
    pts["passage_incoherent"] = pas["incoherent"].reindex(pts["passage"]).to_numpy()
    n_pas_incoh = int(pas["incoherent"].sum())
    pas = pas[~pas["incoherent"]]
    gp = pas.groupby("hk")
    gpt = pts[~pts["passage_incoherent"]].groupby("hk")["E"]
    hexs = pd.DataFrame(dict(
        q=gp["hq"].first(), r=gp["hr"].first(), n_passages=gp.size(),
        E_moy=gp["E"].mean(),                    # chaque passage pèse autant
        E_moy_pts=gpt.mean(),                    # chaque point pèse autant (pour comparaison)
        E_med=gpt.median(), E_p10=gpt.quantile(0.10), E_min=gpt.min(), E_max=gpt.max(),
        n_pts=gpt.size(),
        cv_passages=(gp["E"].std() / gp["E"].mean()),
        h_first=gp["h"].min(), h_last=gp["h"].max(),
    ))
    hexs["U0"] = (hexs["E_p10"] / hexs["E_moy_pts"]).clip(upper=1)
    hexs["conforme"] = hexs["E_moy"] >= P.seuil_emoy_lx
    cx, cy = hex_center(hexs["q"].to_numpy(), hexs["r"].to_numpy(), R)
    hexs["cx"], hexs["cy"] = cx, cy
    multi = hexs[hexs["n_passages"] >= 2]
    T["hexagones"] = _t() - t1

    # ---------------- inventaire
    t1 = _t()
    inv, sup, sheet = load_inventaire(xlsx_path, hloc0)
    inv_syn = synthese_inventaire(inv)
    T["inventaire"] = _t() - t1

    # ---------------- rattachement mesures <-> supports
    t1 = _t()
    tree_p = cKDTree(pts[["x", "y"]].to_numpy())
    Ev = pts["E"].to_numpy()
    runs = pts["run"].to_numpy()
    tt = pts["t"].to_numpy()
    hh = pts["h_loc"].to_numpy()
    L1, L3 = pts["L1"].to_numpy(), pts["L3"].to_numpy()
    PX, PY = pts["x"].to_numpy(), pts["y"].to_numpy()
    capp = np.radians(pts["cap"].to_numpy())
    sup["x_inv"], sup["y_inv"] = sup["x"].to_numpy(), sup["y"].to_numpy()   # position déclarée dans l'Excel

    def mesurer(sup):
        """Mesures au passage autour de chaque support, à partir de sup["x"], sup["y"].
        Appelée deux fois : positions de l'inventaire, puis positions recalées sur les points lumineux."""
        dmin, _ = tree_p.query(sup[["x", "y"]].to_numpy(), k=1)
        sup["dist_min"] = dmin
        near = tree_p.query_ball_point(sup[["x", "y"]].to_numpy(), r=P.rayon_support_m)
        sx, sy = sup["x"].to_numpy(), sup["y"].to_numpy()
        emax, npass, hpass, side_geo, asym, imax_l = [], [], [], [], [], []
        for i, nb_ in enumerate(near):
            if not nb_:
                emax.append(np.nan); npass.append(0); hpass.append(np.nan); side_geo.append(0); asym.append(np.nan)
                imax_l.append(-1)
                continue
            nb_ = np.asarray(nb_)
            # passages distincts = coupure temporelle > 20 s
            o2 = nb_[np.argsort(tt[nb_])]
            cut = np.r_[0, np.cumsum(np.diff(tt[o2]) > P.gap_passage_s)]
            maxs, hs, ims = [], [], []
            for c in np.unique(cut):
                g = o2[cut == c]
                j = g[np.argmax(Ev[g])]
                maxs.append(Ev[j]); hs.append(hh[j]); ims.append(j)
            k = int(np.argsort(maxs)[len(maxs) // 2])  # passage médian (robuste)
            emax.append(float(np.median(maxs)))
            npass.append(len(maxs))
            hpass.append(hs[k])
            j = ims[int(np.argmax(maxs))]
            imax_l.append(j)
            # côté géométrique du support par rapport au sens de marche (au point le plus proche)
            jn = o2[np.argmin(np.hypot(PX[o2] - sx[i], PY[o2] - sy[i]))]
            hx, hy = math.sin(capp[jn]), math.cos(capp[jn])
            cross = hx * (sy[i] - PY[jn]) - hy * (sx[i] - PX[jn])
            side_geo.append(1 if cross > 0 else -1)  # +1 = à gauche
            a = (L1[j] - L3[j]) / (L1[j] + L3[j] + 1)
            asym.append(a)
        sup["E_max_passage"] = emax
        sup["n_passages"] = npass
        sup["h_passage"] = hpass
        sup["cote_geo"] = side_geo
        sup["asym"] = asym
        sup["frole"] = sup["dist_min"] <= P.dist_frole_m
        sup["suspect"] = sup["frole"] & (sup["E_max_passage"] < P.seuil_eteint_lx)
        # calibration du côté des capteurs : Lux1 est-il à gauche ou à droite ?
        m = sup["frole"] & (sup["dist_min"] > 1.5) & sup["asym"].notna() & (sup["asym"].abs() > 0.05) & ~sup["suspect"]
        agree = float((np.sign(sup.loc[m, "asym"]) == sup.loc[m, "cote_geo"]).mean()) if m.any() else 0.5
        lux1_gauche = agree >= 0.5
        side_acc = agree if lux1_gauche else 1 - agree
        return imax_l, lux1_gauche, side_acc

    imax_l, lux1_gauche, side_acc = mesurer(sup)
    T["supports"] = _t() - t1


    # ---------------- détection des luminaires dans le signal (bonus)
    t1 = _t()
    peaks_all = []
    for rid, g in pts.groupby("run"):
        if len(g) < 10:
            continue
        e = g["E"].to_numpy()
        es = uniform_filter1d(e, 3)
        pk, pr = find_peaks(es, prominence=np.maximum(3.0, 0.25 * es.max() * 0), distance=4, height=4)
        prom = pr["prominences"]
        sel = prom >= np.maximum(3.0, 0.3 * es[pk])
        for j, pv in zip(g.index.to_numpy()[pk[sel]], prom[sel]):
            peaks_all.append((j, pv))
    pk_idx = np.array([p[0] for p in peaks_all], int)
    pk_prom = np.array([p[1] for p in peaks_all])
    a = (L1[pk_idx] - L3[pk_idx]) / (L1[pk_idx] + L3[pk_idx] + 1)
    gauche = (a > 0) if lux1_gauche else (a < 0)
    sgn = np.where(gauche, 1, -1)
    hx, hy = np.sin(capp[pk_idx]), np.cos(capp[pk_idx])
    # vecteur gauche = (-hy, hx)
    ex = PX[pk_idx] + sgn * P.decalage_lateral_m * (-hy)
    ey = PY[pk_idx] + sgn * P.decalage_lateral_m * hx
    # regroupement des détections des différents passages (rayon 8 m, glouton par proéminence)
    order = np.argsort(-pk_prom)
    tree_e = cKDTree(np.c_[ex, ey]) if len(ex) else None
    lab = -np.ones(len(ex), int)
    clusters = []
    for i in order:
        if lab[i] >= 0:
            continue
        nb_ = [j for j in tree_e.query_ball_point([ex[i], ey[i]], 8.0) if lab[j] < 0]
        cid = len(clusters)
        lab[nb_] = cid
        nb_ = np.asarray(nb_)
        clusters.append(dict(x=float(ex[nb_].mean()), y=float(ey[nb_].mean()), n=int(len(nb_)),
                             E=float(np.median(Ev[pk_idx[nb_]])), gauche=float(gauche[nb_].mean())))
    det = pd.DataFrame(clusters)

    # ---------------- recalage de l'inventaire sur les points lumineux détectés
    # Les coordonnées de l'Excel sont parfois fausses de plusieurs mètres (vérifié sur Street View).
    # Chaque luminaire détecté est apparié à un seul support (affectation optimale), puis le support
    # est déplacé le long de la voie jusqu'au pic de lumière mesuré (voir recaler_inventaire).
    rec, det_sup, recal_inv = recaler_inventaire(
        sup, PX[pk_idx], PY[pk_idx], np.sin(capp[pk_idx]), np.cos(capp[pk_idx]), sgn, lab, len(det), P, log)
    for c_ in rec.columns:
        sup[c_] = rec[c_].to_numpy()
    if P.recaler_inventaire:
        # supports appariés : position du point lumineux ; autres : position Excel + décalage d'ensemble éventuel
        sup["x"], sup["y"] = sup["x_corr"].to_numpy(), sup["y_corr"].to_numpy()
        imax_l, lux1_gauche, side_acc = mesurer(sup)   # mesures refaites aux positions recalées
    lonc, latc = TO_WGS.transform(sup["x_corr"].to_numpy(), sup["y_corr"].to_numpy())
    sup["LONGITUDE_CORR"], sup["LATITUDE_CORR"] = np.round(lonc, 7), np.round(latc, 7)
    tree_s = cKDTree(sup[["x", "y"]].to_numpy())

    if len(det):
        dd, ii = tree_s.query(det[["x", "y"]].to_numpy(), k=1)
        det["dist_support"] = dd
        det["support"] = sup["NUMERO"].to_numpy()[ii]
        det["apparie"] = det_sup >= 0
        det.loc[det["apparie"], "support"] = sup["NUMERO"].to_numpy()[det_sup[det_sup >= 0]]
        # un pic non affecté mais à moins de rayon_match/2 d'un support = 2e luminaire du même support
        det["proche_support"] = dd <= P.rayon_match_m / 2
    else:
        det = pd.DataFrame(columns=["x", "y", "n", "E", "gauche", "dist_support", "support", "apparie",
                                    "proche_support"])
    # détecté = apparié à un pic, ou pic à moins de rayon_match/2 de la position recalée (pic commun à 2 supports voisins)
    if len(det):
        dd2, _ = cKDTree(det[["x", "y"]].to_numpy()).query(sup[["x", "y"]].to_numpy(), k=1)
    else:
        dd2 = np.full(len(sup), np.inf)
    sup["detecte"] = sup["apparie_pic"].to_numpy() | (dd2 <= P.rayon_match_m / 2)
    ref = sup["frole"] & ~sup["suspect"]
    taux_detection = float(sup.loc[ref, "detecte"].mean()) if ref.any() else None
    autres = load_autres_sources(xlsx_path)
    if len(det) and len(autres):
        da, _ = cKDTree(autres).query(det[["x", "y"]].to_numpy(), k=1)
        det["autre_source"] = da <= P.rayon_match_m
    else:
        det["autre_source"] = False
    non_inv = det[(~det["apparie"].astype(bool)) & (~det["proche_support"].astype(bool))
                  & (~det["autre_source"].astype(bool)) & (det["n"] >= 2) & (det["E"] >= 10)] if len(det) else det
    T["detection"] = _t() - t1

    # ---------------- chromamètre : source mesurée au passage
    # Constat sur les données : le chromamètre renvoie Tcp = 0 quand la température de couleur est
    # hors de sa plage (< ~2200 K), ce qui est le cas du sodium haute pression (~2000 K).
    # Règle : au passage (±2 s autour du maximum, Ev > 5 lx), si la majorité des mesures sont à 0
    # -> source « sodium/ambrée », sinon -> source « blanche » avec la Tcp médiane mesurée.
    t1 = _t()
    sup["cct"] = np.nan
    sup["part_cct0"] = np.nan
    if cal is not None:
        calv = cal[cal["Ev"] > 5]
        tc, tcp = calv["t"].to_numpy(), calv["Tcp"].to_numpy()
        for i, j in enumerate(imax_l):
            if j < 0 or not sup.at[i, "frole"]:
                continue
            a_, b_ = np.searchsorted(tc, tt[j] - 2), np.searchsorted(tc, tt[j] + 2)
            if b_ - a_ < 2:
                continue
            w_ = tcp[a_:b_]
            sup.at[i, "part_cct0"] = float((w_ == 0).mean())
            if (w_ > 0).any():
                sup.at[i, "cct"] = float(np.median(w_[w_ > 0]))
    pz = sup["part_cct0"]
    sup["source_mesuree"] = np.where(pz.isna(), None, np.where(pz >= 0.5, "sodium", "blanche"))
    fam_ = sup["famille"].astype(str)
    decl = np.where(fam_.str.contains("SODIUM HAUTE PRESSION") & ~fam_.str.contains("BLANC"), "sodium",
                    np.where(fam_.isin(["LED", "COSMOWHITE", "IODURE METALLIQUE"]), "blanche", None))
    sup["source_declaree"] = decl
    sure = (pz == 0) | (pz == 1)   # tous les échantillons du passage concordent
    sup["incoherence_source"] = (sup["source_mesuree"].notna() & pd.Series(decl).notna().to_numpy()
                                 & (sup["source_mesuree"] != sup["source_declaree"]) & sure
                                 & ~sup["modif_apres_releve"].astype(bool)
                                 & ~(sup["k_decl"] <= 2200).fillna(False).to_numpy())
    # validation : luminaires LED posés APRÈS le relevé -> devaient encore être en sodium ce soir-là
    lv = sup[(fam_ == "LED") & sup["modif_apres_releve"].astype(bool) & sup["source_mesuree"].notna()]
    lb = sup[(fam_ == "LED") & ~sup["modif_apres_releve"].astype(bool) & sup["source_mesuree"].notna()]
    ls = sup[(decl == "sodium") & ~sup["modif_apres_releve"].astype(bool) & sup["source_mesuree"].notna()]
    chroma = dict(
        n=int(sup["source_mesuree"].notna().sum()),
        led_apres_n=int(len(lv)), led_apres_sodium=_r(100 * (lv["source_mesuree"] == "sodium").mean(), 0) if len(lv) else None,
        led_avant_n=int(len(lb)), led_avant_blanche=_r(100 * (lb["source_mesuree"] == "blanche").mean(), 0) if len(lb) else None,
        sodium_n=int(len(ls)), sodium_ok=_r(100 * (ls["source_mesuree"] == "sodium").mean(), 0) if len(ls) else None,
    )
    T["chromametre"] = _t() - t1

    # ---------------- rues (rattachement au support le plus proche)
    t1 = _t()
    ds_, is_ = tree_s.query(pts[["x", "y"]].to_numpy(), k=1, distance_upper_bound=P.rayon_rue_m)
    okr = np.isfinite(ds_)
    pts["rue"] = None
    pts.loc[okr, "rue"] = sup["ZONE"].to_numpy()[is_[okr]]
    pts["armoire"] = None
    pts.loc[okr, "armoire"] = sup["PARENT"].to_numpy()[is_[okr]]
    # sens de passage : pour chaque rue, axe principal = moyenne axiale des caps (angles doublés),
    # puis sens 1 si le cap est à moins de 90° de cet axe, sens 2 sinon. Hors rue : axe global.
    a2 = np.radians(pts["cap"].to_numpy())
    key_r = pts["rue"].fillna("__hors_rue__")
    s2 = pd.Series(np.sin(2 * a2)).groupby(key_r.to_numpy()).transform("mean").to_numpy()
    c2 = pd.Series(np.cos(2 * a2)).groupby(key_r.to_numpy()).transform("mean").to_numpy()
    axe = 0.5 * np.arctan2(s2, c2)
    pts["sens"] = np.where(np.cos(a2 - axe) >= 0, 1, 2)
    rg = pts[okr].groupby("rue")
    rues = pd.DataFrame(dict(n=rg.size(), E_moy=rg["E"].mean(), E_med=rg["E"].median(),
                             E_p10=rg["E"].quantile(0.10), n_runs=rg["run"].nunique()))
    rues["longueur_m"] = rues["n"] * P.pas_m
    rues["U"] = rues["E_p10"] / rues["E_moy"]
    rues = rues[rues["n"] >= 10].sort_values("E_moy")
    ss = sup.groupby("ZONE").agg(n_sup=("ID", "size"), n_frole=("frole", "sum"), n_suspects=("suspect", "sum"))
    rues = rues.join(ss, how="left")
    rues["conforme"] = rues["E_moy"] >= P.seuil_emoy_lx
    T["rues"] = _t() - t1

    # ---------------- armoires : circuits possiblement coupés
    arm = sup[sup["frole"]].groupby("PARENT").agg(n_frole=("ID", "size"), n_suspects=("suspect", "sum"),
                                                  E_med=("E_max_passage", "median"))
    arm["part"] = arm["n_suspects"] / arm["n_frole"]
    arm_alerte = arm[(arm["n_suspects"] >= 3) & (arm["part"] >= 0.3)].sort_values("part", ascending=False)

    # ---------------- abaissement nocturne : cellules passées avant ET après l'heure d'abaissement
    t1 = _t()
    var_ids = sup["regime"].astype(str).str.upper().str.startswith("VARIATION")
    tv = cKDTree(sup.loc[var_ids, ["x", "y"]].to_numpy()) if var_ids.any() else None
    pas = pas.join(hexs[["cx", "cy"]], on="hk")
    pas["avant"] = pas["h"] < P.heure_abaissement
    both = pas.groupby("hk")["avant"].agg(["any", "all"])
    both = both[both["any"] & ~both["all"]].index
    pb = pas[pas["hk"].isin(both)]
    piv = pb.groupby(["hk", "avant"])["E"].mean().unstack()
    piv.columns = ["apres" if not c else "avant" for c in piv.columns]
    piv = piv.join(hexs[["cx", "cy"]])
    if tv is not None and len(piv):
        dv, _ = tv.query(piv[["cx", "cy"]].to_numpy(), k=1)
        piv["variation"] = dv <= 25
    else:
        piv["variation"] = False
    piv = piv[(piv["avant"] > 2)]
    piv["ratio"] = piv["apres"] / piv["avant"]
    abaiss = {}
    for lab_, g in [("variation", piv[piv["variation"]]), ("autres", piv[~piv["variation"]])]:
        abaiss[lab_] = dict(n=int(len(g)), ratio_med=_r(g["ratio"].median(), 2),
                            avant=_r(g["avant"].median()), apres=_r(g["apres"].median()))
    T["abaissement"] = _t() - t1

    # ---------------- séries pour les graphiques
    hist_edges = np.arange(0, 102.5, 2.5)
    hcount, _ = np.histogram(np.clip(pts["E"], 0, 100), bins=hist_edges)
    tsm = pts.assign(m=(pts["h_loc"] * 60).round()).groupby("m")["E"].median()
    tsm = tsm.rolling(5, center=True, min_periods=1).median()
    fam_stats = []
    for f, g in sup[sup["frole"] & sup["E_max_passage"].notna()].groupby("famille"):
        if len(g) < 5:
            continue
        v = g["E_max_passage"].to_numpy()
        fam_stats.append(dict(f=f, n=int(len(g)), q=[_r(np.percentile(v, p)) for p in (10, 25, 50, 75, 90)],
                              P=_r(g["puissance_w"].mean(), 0)))
    fam_stats.sort(key=lambda d_: -d_["q"][2])

    # ---------------- résumé
    hc = hexs
    res = dict(
        meta=dict(
            fichier_tdms=Path(tdms_path).name, fichier_xlsx=Path(xlsx_path).name, onglet=sheet,
            ville=guess_ville(tdms_path, inv), date=hloc0.strftime("%d/%m/%Y"),
            date_iso=hloc0.strftime("%Y-%m-%d"),
            debut=hloc0.strftime("%Hh%M"), fin=hloc1.strftime("%Hh%M"),
            genere=datetime.now(TZ).strftime("%d/%m/%Y %Hh%M"), params=asdict(P),
        ),
        kpi=dict(
            dist_km=_r(dist_km, 1), duree_h=qual["duree_h"], E_med=_r(pts["E"].median()),
            E_moy=_r(pts["E"].mean()), E_p25=_r(pts["E"].quantile(.25)), E_p75=_r(pts["E"].quantile(.75)),
            part_sup50=_r(100 * (pts["E"] > 50).mean(), 1),
            n_sup=int(len(sup)), n_frole=int(sup["frole"].sum()), part_frole=_r(100 * sup["frole"].mean(), 0),
            n_suspects=int(sup["suspect"].sum()), n_hex=int(len(hc)), n_hex_multi=int(len(multi)),
            part_hex_conf=_r(100 * hc["conforme"].mean(), 0),
            cv_multi_med=_r(100 * multi["cv_passages"].median(), 0) if len(multi) else None,
            n_detect=int(len(det)), taux_detection=_r(100 * taux_detection, 0) if taux_detection is not None else None,
            n_non_inv=int(len(non_inv)), lux1_gauche=bool(lux1_gauche), precision_cote=_r(100 * side_acc, 0),
            n_incoh_source=int(sup["incoherence_source"].sum()),
            n_cct=int(sup["cct"].notna().sum()),
            n_pas_incoh=n_pas_incoh, n_aberrants=bruit["n_trop_haut"] + bruit["n_trop_bref"] + bruit["n_arret"],
            n_rues=int(len(rues)), n_rues_conf=int(rues["conforme"].sum()),
            n_pos_corr=int(sup["position_douteuse"].sum()),
        ),
        qualite=qual, methodes=stats_methodes, chroma=chroma, calage=calage, recalage_inventaire=recal_inv, inventaire=inv_syn, abaissement=abaiss,
        hist=[[float(e), int(c)] for e, c in zip(hist_edges[:-1], hcount)],
        ts=[[_r(m_ / 60, 3), _r(v, 1)] for m_, v in tsm.items()],
        fam=fam_stats,
        armoires=[[k, int(v.n_frole), int(v.n_suspects), _r(v.E_med)] for k, v in arm_alerte.iterrows()][:12],
        rues=[[k, int(v.n), _r(v.E_moy), _r(v.E_med), _r(v.E_p10), _r(v.U, 2), int(v.n_runs),
               int(v.n_sup) if pd.notna(v.n_sup) else 0, int(v.n_suspects) if pd.notna(v.n_suspects) else 0]
              for k, v in rues.iterrows()],
        suspects=[[s.NUMERO, s.ZONE, str(s.famille).lower(), _r(s.puissance_w, 0), _r(s.E_max_passage),
                   _r(s.dist_min), int(s.n_passages), bool(s.modif_apres_releve), s.PARENT]
                  for s in sup[sup["suspect"]].sort_values(["ZONE", "NUMERO"]).itertuples()],
        incoherences=[[s.NUMERO, s.ZONE, str(s.famille).lower(), _r(s.cct, 0), s.source_mesuree, s.PARENT]
                      for s in sup[sup["incoherence_source"]].sort_values(["ZONE", "NUMERO"]).itertuples()],
        positions=[[s.NUMERO, s.ZONE, _r(s.ecart_position_m), _r(s.decalage_voie_m), bool(s.cote_inverse),
                    int(s.n_pics), _r(s.dispersion_m), round(s.LATITUDE, 7), round(s.LONGITUDE, 7),
                    s.LATITUDE_CORR, s.LONGITUDE_CORR, s.PARENT]
                   for s in sup[sup["position_douteuse"]].sort_values("ecart_position_m", ascending=False).itertuples()],
        temps={k: _r(v, 2) for k, v in T.items()},
    )
    # ---- données pour la carte (coordonnées locales en m, origine = centre)
    ox, oy = float(pts["x"].median()), float(pts["y"].median())
    res["carte"] = carte_data(pts, hexs, sup, non_inv, R, lux1_gauche)
    res["map"] = dict(
        origin=[ox, oy],
        P=[[_r(a_ - ox, 1), _r(b_ - oy, 1), _r(e_, 1), _r(h_, 3)]
           for a_, b_, e_, h_ in pts[["x", "y", "E", "h_loc"]].to_numpy()[::2]],
        S=[[_r(s.x - ox, 1), _r(s.y - oy, 1), int(s.fam3), _r(s.E_max_passage), s.NUMERO, s.ZONE,
            str(s.famille).lower(), _r(s.puissance_w, 0), _r(s.dist_min, 0), int(bool(s.suspect)),
            int(bool(s.incoherence_source)), int(s.n_passages), int(bool(s.position_douteuse)),
            _r(s.x_inv - ox, 1), _r(s.y_inv - oy, 1), _r(s.ecart_position_m)]
           for s in sup.itertuples()],
        H=[[_r(h.cx - ox, 1), _r(h.cy - oy, 1), _r(h.E_moy), int(h.n_passages), _r(h.U0, 2), _r(h.E_min)]
           for h in hexs.itertuples()],
        D=[[_r(d_.x - ox, 1), _r(d_.y - oy, 1), _r(d_.E), int(d_.n)] for d_ in non_inv.itertuples()],
        R=P.hex_m,
    )

    # ---------------- exports
    t1 = _t()
    exports(out_dir, pts, hexs, sup, rues, det, R)
    T["exports"] = _t() - t1
    res["temps"] = {k: _r(v, 2) for k, v in T.items()}
    res["temps"]["total"] = _r(_t() - t0, 1)
    (out_dir / "resume.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("map", "carte")},
                                                    ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


def load_autres_sources(xlsx):
    """Coordonnées des autres ouvrages lumineux connus (éclairage sportif, mobilier urbain alimenté)."""
    out = []
    try:
        xl = pd.ExcelFile(xlsx)
        for sh in xl.sheet_names:
            u = sh.upper()
            if ("SPORTIF" in u or "MOBILIER" in u):
                d = xl.parse(sh, header=1)
                if "LATITUDE" in d:
                    d = d[d["LATITUDE"].notna() & d["LONGITUDE"].notna()]
                    x, y = TO_L93.transform(d["LONGITUDE"].to_numpy(float), d["LATITUDE"].to_numpy(float))
                    out += list(zip(x, y))
    except Exception:
        pass
    return np.array(out).reshape(-1, 2)


def guess_ville(tdms_path, inv):
    v = inv["VILLE"].dropna()
    if len(v):
        s = str(v.mode().iloc[0]).title().replace(" Sur ", "-sur-").replace(" ", "-")
        return s
    return Path(tdms_path).stem


def carte_data(pts, hexs, sup, non_inv, R, lux1_gauche):
    """Données compactes pour la carte satellite (WGS84)."""
    lon, lat = TO_WGS.transform(pts["x"].to_numpy(), pts["y"].to_numpy())
    # capteurs gauche / centre / droite dans l'ordre de la chaussée
    g, d = ("L1", "L3") if lux1_gauche else ("L3", "L1")
    P = [[round(a, 6), round(b, 6), _r(e, 1), _r(lg, 1), _r(lc, 1), _r(ld, 1), int(sn), _r(cp, 0), _r(h, 3)]
         for a, b, e, lg, lc, ld, sn, cp, h in zip(lon, lat, pts["E"], pts[g], pts["L2"], pts[d], pts["sens"],
                                                  pts["cap"], pts["h_loc"])]
    H = []
    for h in hexs.itertuples():
        poly = hex_polygon(h.cx, h.cy, R)
        ln, lt = TO_WGS.transform([q[0] for q in poly], [q[1] for q in poly])
        H.append([[[round(b, 6), round(a, 6)] for a, b in zip(ln, lt)], _r(h.E_moy), _r(h.E_min), _r(h.U0, 2),
                  int(h.n_passages)])
    # position affichée = position recalée sur le point lumineux ; [8], [9] = position de l'Excel
    S = [[round(s.LATITUDE_CORR, 6), round(s.LONGITUDE_CORR, 6), s.NUMERO, s.ZONE, str(s.famille).lower(),
          _r(s.E_max_passage), int(bool(s.suspect)), int(s.fam3), round(s.LATITUDE, 6), round(s.LONGITUDE, 6),
          _r(s.ecart_position_m), int(bool(s.position_douteuse)), int(bool(s.recalage_fiable))]
         for s in sup.itertuples()]
    Dn = []
    if len(non_inv):
        ln, lt = TO_WGS.transform(non_inv["x"].to_numpy(), non_inv["y"].to_numpy())
        Dn = [[round(b, 6), round(a, 6), _r(e, 1), int(n)] for a, b, e, n in zip(ln, lt, non_inv["E"], non_inv["n"])]
    return dict(P=P, H=H, S=S, D=Dn, capteurs=["gauche (" + g.replace("L", "Lux") + ")", "centre (Lux2)",
                                                  "droite (" + d.replace("L", "Lux") + ")"])


def exports(out_dir: Path, pts, hexs, sup, rues, det, R):
    lon, lat = TO_WGS.transform(pts["x"].to_numpy(), pts["y"].to_numpy())
    e = pts.assign(lon=lon, lat=lat)
    e["heure_locale"] = e["h_loc"].round(4)
    e[["lon", "lat", "x", "y", "E", "L1", "L2", "L3", "Emax_brut", "v", "cap", "sens", "heure_locale", "rue", "passage"]] \
        .round(dict(lon=7, lat=7, x=2, y=2, E=2, L1=2, L2=2, L3=2, Emax_brut=2, v=2, cap=1)) \
        .to_csv(out_dir / "mesures_pas2m.csv", index=False, sep=";", decimal=",")
    feats = []
    for h in hexs.itertuples():
        poly = hex_polygon(h.cx, h.cy, R)
        ln, lt = TO_WGS.transform([p[0] for p in poly], [p[1] for p in poly])
        ring = [[round(a, 7), round(b, 7)] for a, b in zip(ln, lt)]
        ring.append(ring[0])
        feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[ring]), properties=dict(
            E_moy=_r(h.E_moy, 2), E_med=_r(h.E_med, 2), E_p10=_r(h.E_p10, 2), E_min=_r(h.E_min, 2),
            E_max=_r(h.E_max, 2), U0=_r(h.U0, 3), n_passages=int(h.n_passages), n_pts=int(h.n_pts),
            cv_passages=_r(h.cv_passages, 3), conforme=bool(h.conforme))))
    (out_dir / "hexagones.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats)),
                                               encoding="utf-8")
    cols = ["ID", "NUMERO", "ZONE", "PARENT", "LATITUDE", "LONGITUDE", "famille", "puissance_w", "etat", "regime",
            "dist_min", "n_passages", "E_max_passage", "h_passage", "frole", "suspect", "cct", "source_mesuree",
            "source_declaree", "part_cct0", "incoherence_source", "detecte", "modif_apres_releve",
            "LATITUDE_CORR", "LONGITUDE_CORR", "ecart_position_m", "decalage_voie_m", "cote_inverse", "n_pics",
            "dispersion_m", "recalage_fiable", "position_douteuse"]
    rnd = dict(dist_min=1, E_max_passage=1, h_passage=3, cct=0, ecart_position_m=1, decalage_voie_m=1, dispersion_m=1)
    sup[cols].round(rnd).to_csv(out_dir / "supports.csv", index=False, sep=";", decimal=",")
    # fichier prêt pour la mise à jour du SIG : uniquement les positions à corriger
    pc = sup[sup["position_douteuse"]].sort_values("ecart_position_m", ascending=False)
    pc[["ID", "NUMERO", "ZONE", "PARENT", "LATITUDE", "LONGITUDE", "LATITUDE_CORR", "LONGITUDE_CORR",
        "ecart_position_m", "decalage_voie_m", "cote_inverse", "n_pics", "dispersion_m"]].round(rnd) \
        .to_csv(out_dir / "positions_a_corriger.csv", index=False, sep=";", decimal=",")
    rues.round(2).to_csv(out_dir / "rues.csv", sep=";", decimal=",", index_label="rue")
    if len(det):
        ln, lt = TO_WGS.transform(det["x"].to_numpy(), det["y"].to_numpy())
        feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[round(a, 7), round(b, 7)]),
                      properties=dict(E_pic=_r(r.E, 1), n_detections=int(r.n), part_gauche=_r(r.gauche, 2),
                                      support_proche=r.support, dist_support=_r(r.dist_support, 1),
                                      apparie=bool(r.apparie)))
                 for a, b, r in zip(ln, lt, det.itertuples())]
        (out_dir / "luminaires_detectes.geojson").write_text(
            json.dumps(dict(type="FeatureCollection", features=feats)), encoding="utf-8")
