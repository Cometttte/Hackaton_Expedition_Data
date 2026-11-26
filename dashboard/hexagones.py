"""Statistiques par cellule hexagonale (1 m par défaut), calculées à partir de mesures_pas2m.csv.

Même règle que pour les tronçons :
  * un passage = suite continue de points dans la même cellule (coupure si > 20 s)
  * moyenne d'un sens = moyenne des moyennes de ses passages (chaque passage pèse autant)
  * E_moy de la cellule = moyenne des deux sens (colonne `sens` de l'analyse : 1 / 2 sur l'axe local
    de la voie) ; un seul sens mesuré -> la valeur de ce sens
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pyproj import Transformer

TAILLE_M = 1.0        # hexagone « pointe en haut » de 1 m entre côtés opposés
GAP_S = 20.0
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


def calculer(m: pd.DataFrame, taille_m: float = TAILLE_M):
    """m : mesures avec x, y (L93), lux, heure_locale et, si présent, sens.
    Retourne (GeoJSON FeatureCollection, DataFrame des propriétés)."""
    R = taille_m / math.sqrt(3)                      # rayon centre-sommet
    q, r = _cellule(m["x"].to_numpy(), m["y"].to_numpy(), R)
    df = pd.DataFrame({"q": q, "r": r, "lux": m["lux"].to_numpy(),
                       "h": m["heure_locale"].to_numpy() * 3600,
                       "sens": m["sens"].to_numpy() if "sens" in m else np.ones(len(m), int)})
    cle = df["q"].to_numpy() * 10_000_000 + df["r"].to_numpy()
    newp = np.r_[True, (cle[1:] != cle[:-1]) | (np.diff(df["h"].to_numpy()) > GAP_S)]
    df["passage"] = np.cumsum(newp)

    pas = df.groupby(["q", "r", "passage"]).agg(Ep=("lux", "mean"), sens=("sens", "first")).reset_index()
    sens = pas.groupby(["q", "r", "sens"])["Ep"].mean().unstack("sens")
    e1 = sens.get(1, pd.Series(np.nan, index=sens.index))
    e2 = sens.get(2, pd.Series(np.nan, index=sens.index))
    gp = pas.groupby(["q", "r"])["Ep"]
    gpts = df.groupby(["q", "r"])["lux"]
    st = pd.DataFrame({
        "E_moy": pd.concat([e1, e2], axis=1).mean(axis=1), "E_sens1": e1, "E_sens2": e2,
        "n_sens": e1.notna().astype(int) + e2.notna().astype(int),
        "E_med": gpts.median(), "E_min": gpts.min(), "E_max": gpts.max(),
        "n_passages": gp.size(), "n_pts": gpts.size(),
    }).reset_index()
    st["U0"] = (st["E_min"] / st["E_moy"]).where(st["E_moy"] > 0, 0)

    # sommets de tous les hexagones d'un coup (vectorisé)
    cx = R * math.sqrt(3) * (st["q"].to_numpy() + st["r"].to_numpy() / 2)
    cy = R * 1.5 * st["r"].to_numpy()
    ang = np.radians(60 * np.arange(7) - 30)
    vx = (cx[:, None] + R * np.cos(ang)[None, :]).ravel()
    vy = (cy[:, None] + R * np.sin(ang)[None, :]).ravel()
    vlon, vlat = _to84.transform(vx, vy)
    vlon, vlat = np.round(vlon, 7).reshape(-1, 7), np.round(vlat, 7).reshape(-1, 7)
    clon, clat = _to84.transform(cx, cy)

    feats = [{"type": "Feature", "id": k,
              "geometry": {"type": "Polygon", "coordinates": [np.c_[vlon[k], vlat[k]].tolist()]}}
             for k in range(len(st))]
    props = st.assign(_id=np.arange(len(st)), _lon=clon, _lat=clat).round(
        {"E_moy": 2, "E_sens1": 2, "E_sens2": 2, "E_med": 2, "U0": 3})
    return {"type": "FeatureCollection", "features": feats}, props
