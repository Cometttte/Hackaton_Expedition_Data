"""Statistiques par cellule hexagonale (10 m), calculées à partir de mesures_pas2m.csv.

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

RAYON_M = 10.0 / math.sqrt(3)     # hexagone « pointe en haut » de 10 m entre côtés opposés
GAP_S = 20.0
_to84 = Transformer.from_crs(2154, 4326, always_xy=True)


def _cellule(x, y, R=RAYON_M):
    """Coordonnées axiales (q, r) de l'hexagone contenant chaque point (arrondi cubique)."""
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
    return rx.astype(int), rz.astype(int)


def calculer(m: pd.DataFrame):
    """m : mesures avec x, y (L93), lux, heure_locale et, si présent, sens.
    Retourne (GeoJSON FeatureCollection, DataFrame des propriétés)."""
    q, r = _cellule(m["x"].to_numpy(), m["y"].to_numpy())
    df = pd.DataFrame({"q": q, "r": r, "lux": m["lux"].to_numpy(),
                       "h": m["heure_locale"].to_numpy() * 3600,
                       "sens": m["sens"].to_numpy() if "sens" in m else np.ones(len(m), int)})
    cle = df["q"].to_numpy() * 100_000 + df["r"].to_numpy()
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
        "E_med": gpts.median(), "E_p10": gpts.quantile(0.10), "E_min": gpts.min(), "E_max": gpts.max(),
        "n_passages": gp.size(), "n_pts": gpts.size(),
        "cv_passages": (gp.std() / gp.mean()).fillna(0),
    }).reset_index()
    st["U0"] = (st["E_min"] / st["E_moy"]).where(st["E_moy"] > 0, 0)

    R = RAYON_M
    ang = np.radians(60 * np.arange(7) - 30)
    feats, props = [], []
    for k, s in enumerate(st.itertuples()):
        cx = R * math.sqrt(3) * (s.q + s.r / 2)
        cy = R * 1.5 * s.r
        lon, lat = _to84.transform(cx + R * np.cos(ang), cy + R * np.sin(ang))
        clon, clat = _to84.transform(cx, cy)
        p = {"id": k, "E_moy": round(s.E_moy, 2), "E_med": round(s.E_med, 2), "E_p10": round(s.E_p10, 2),
             "E_sens1": None if pd.isna(s.E_sens1) else round(s.E_sens1, 2),
             "E_sens2": None if pd.isna(s.E_sens2) else round(s.E_sens2, 2),
             "n_sens": int(s.n_sens), "U0": round(s.U0, 3), "n_passages": int(s.n_passages),
             "n_pts": int(s.n_pts), "cv_passages": round(s.cv_passages, 3)}
        feats.append({"type": "Feature", "id": k, "properties": p,
                      "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in zip(lon, lat)]]}})
        props.append({**p, "_id": k, "_lon": clon, "_lat": clat})
    return {"type": "FeatureCollection", "features": feats}, pd.DataFrame(props)
