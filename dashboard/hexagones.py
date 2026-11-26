"""Cellules hexagonales de 2 m (comme dans la chaîne GETYS : bande continue d'hexagones le long du trajet).

Construction des points (comme getys_spatialisation.py : un point par mesure et par luxmètre) :
  * chaque mesure de mesures_pas2m.csv donne 3 points, un par luxmètre, décalés latéralement selon leur
    position sur la remorque (propriété lux_position du TDMS : Lux1 à droite, Lux2 au centre, Lux3 à
    gauche, à 0,49 m de l'axe), dans la direction du cap ;
  * entre deux mesures consécutives d'un même passage (≤ 4,5 m et ≤ 5 s d'écart), on interpole tous les
    0,5 m : les cellules se suivent sans trou le long de la voie.

Statistiques, même règle que pour les tronçons :
  * un passage = une traversée continue de la cellule ;
  * moyenne d'un sens = moyenne des moyennes de ses passages (chaque passage pèse autant) ;
  * E_moy de la cellule = moyenne des deux sens (colonne `sens` : 1 / 2 sur l'axe local de la voie).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pyproj import Transformer

TAILLE_M = 2.0              # hexagone « pointe en haut » de 2 m entre côtés opposés
PAS_INTERP_M = 0.5          # interpolation le long du trajet
LATERAL_M = {"L1": -0.49, "L2": 0.0, "L3": 0.49}    # lux_position du TDMS d'Ormesson (y > 0 = gauche)
SAUT_MAX_M, SAUT_MAX_S = 4.5, 5.0
_to84 = Transformer.from_crs(2154, 4326, always_xy=True)


def _cellule(x, y, R):
    """Coordonnées axiales (q, r) de l'hexagone de rayon R contenant chaque point (arrondi cubique)."""
    q = (math.sqrt(3) / 3 * x - y / 3) / R
    r = (2 / 3 * y) / R
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    a = (dx > dy) & (dx > dz)
    b = ~a & (dy > dz)
    rx = np.where(a, -ry - rz, rx)
    rz = np.where(~a & ~b, -rx - ry, rz)
    return rx.astype(np.int64), rz.astype(np.int64)


def _densifier(m: pd.DataFrame, cols):
    """Points interpolés tous les PAS_INTERP_M entre mesures consécutives d'un même passage."""
    x, y = m["x"].to_numpy(), m["y"].to_numpy()
    t = m["heure_locale"].to_numpy() * 3600
    d = np.hypot(np.diff(x), np.diff(y))
    lie = (d <= SAUT_MAX_M) & (np.abs(np.diff(t)) <= SAUT_MAX_S)
    run = np.r_[0, np.cumsum(~lie)]                     # identifiant de passage continu
    n_seg = np.where(lie, np.maximum(1, np.ceil(d / PAS_INTERP_M)).astype(int), 1)
    i0 = np.repeat(np.arange(len(m) - 1), n_seg)        # point de départ de chaque échantillon
    f = np.concatenate([np.arange(k) / k for k in n_seg]) if len(n_seg) else np.array([])
    f = np.where(lie[i0], f, 0.0)                       # pas de pont entre deux passages
    i0 = np.r_[i0, len(m) - 1]
    f = np.r_[f, 0.0]
    i1 = np.minimum(i0 + 1, len(m) - 1)
    out = {"x": x[i0] + f * (x[i1] - x[i0]), "y": y[i0] + f * (y[i1] - y[i0]), "run": run[i0]}
    cap = np.radians(m["cap"].to_numpy() if "cap" in m else np.zeros(len(m)))
    s, c = np.sin(cap), np.cos(cap)                     # cap interpolé sur le cercle
    out["cap"] = np.arctan2(s[i0] + f * (s[i1] - s[i0]), c[i0] + f * (c[i1] - c[i0]))
    out["sens"] = (m["sens"].to_numpy() if "sens" in m else np.ones(len(m), int))[i0]
    for col in cols:
        v = m[col].to_numpy(float)
        out[col] = v[i0] + f * (v[i1] - v[i0])
    return pd.DataFrame(out)


def calculer(m: pd.DataFrame, taille_m: float = TAILLE_M, par_capteur: bool = True):
    """m : mesures (x, y en L93, lux, cap, sens, heure_locale, et L1/L2/L3 si par_capteur).
    Retourne (GeoJSON FeatureCollection, DataFrame des propriétés)."""
    capteurs = [c for c in LATERAL_M if c in m] if par_capteur else []
    cols = capteurs or ["lux"]
    pts = _densifier(m, cols)
    # un point par luxmètre, décalé vers la gauche (+) ou la droite (-) du véhicule
    gx, gy = -np.cos(pts["cap"].to_numpy()), np.sin(pts["cap"].to_numpy())   # vecteur « gauche » (L93)
    morceaux = []
    for col in cols:
        dl = LATERAL_M.get(col, 0.0) if capteurs else 0.0
        morceaux.append(pd.DataFrame({"x": pts["x"] + dl * gx, "y": pts["y"] + dl * gy, "lux": pts[col],
                                      "run": pts["run"], "sens": pts["sens"]}))
    df = pd.concat(morceaux, ignore_index=True)

    R = taille_m / math.sqrt(3)                         # rayon centre-sommet
    df["q"], df["r"] = _cellule(df["x"].to_numpy(), df["y"].to_numpy(), R)
    pas = df.groupby(["q", "r", "run"]).agg(Ep=("lux", "mean"), sens=("sens", "first")).reset_index()
    sens = pas.groupby(["q", "r", "sens"])["Ep"].mean().unstack("sens")
    e1 = sens.get(1, pd.Series(np.nan, index=sens.index))
    e2 = sens.get(2, pd.Series(np.nan, index=sens.index))
    st = pd.DataFrame({
        "E_moy": pd.concat([e1, e2], axis=1).mean(axis=1), "E_sens1": e1, "E_sens2": e2,
        "n_sens": e1.notna().astype(int) + e2.notna().astype(int),
        "n_passages": pas.groupby(["q", "r"]).size(),
    }).reset_index()

    # sommets de tous les hexagones d'un coup (vectorisé)
    cx = R * math.sqrt(3) * (st["q"].to_numpy() + st["r"].to_numpy() / 2)
    cy = R * 1.5 * st["r"].to_numpy()
    ang = np.radians(60 * np.arange(7) - 30)
    vlon, vlat = _to84.transform((cx[:, None] + R * np.cos(ang)[None, :]).ravel(),
                                 (cy[:, None] + R * np.sin(ang)[None, :]).ravel())
    vlon, vlat = np.round(vlon, 6).reshape(-1, 7), np.round(vlat, 6).reshape(-1, 7)   # 0,1 m près
    clon, clat = _to84.transform(cx, cy)
    feats = [{"type": "Feature", "id": k,
              "geometry": {"type": "Polygon", "coordinates": [np.c_[vlon[k], vlat[k]].tolist()]}}
             for k in range(len(st))]
    props = st.assign(_id=np.arange(len(st)), _lon=clon, _lat=clat).round(
        {"E_moy": 1, "E_sens1": 1, "E_sens2": 1})
    return {"type": "FeatureCollection", "features": feats}, props
