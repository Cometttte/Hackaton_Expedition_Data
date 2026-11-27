"""Heatmap vectorielle de l'éclairement (densité de flux lumineux : 1 lx = 1 lm/m²), au format GeoJSON.

Script autonome, hors dashboard :

    python dashboard/heatmap_flux.py rapports/<dossier_du_rapport> [--res-m 2] [--sigma-m 4] [--rayon-m 6] [--brut]

Chaîne :
  1. Cellules de 2 m (hexagones.py, même règle que le dashboard) : moyenne par passage, puis moyenne
     de chaque sens, puis E_moy = moyenne des deux sens (un seul sens mesuré : sa moyenne).
  2. Surface continue : grille de --res-m, lissage gaussien normalisé (--sigma-m) des E_moy des cellules,
     chaque cellule pesant autant. La surface est coupée à --rayon-m de la cellule mesurée la plus proche :
     pas d'extrapolation loin du trajet.
  3. Isobandes : la surface est découpée selon les classes d'éclairement du dashboard (0–2, 2–5, … > 30 lx),
     et chaque classe devient un MultiPolygon (zones d'un seul tenant, trous compris).
  4. Rendu en dégradé : la même surface, rééchantillonnée en Web Mercator (pixels de --image-m), colorée sur
     une rampe continue de 0 à 50 lx (palette du dashboard), bords estompés en transparence.

Sorties, dans le dossier du rapport :
  heatmap_flux.geojson          isobandes : un MultiPolygon par classe (classe, E_min, E_max, couleur, surface_m2)
  heatmap_flux_points.geojson   centres des cellules de 2 m : E_moy, E_sens1, E_sens2, n_sens, n_passages
  heatmap_flux.png              image en dégradé (RGBA, Web Mercator ; coins dans la page HTML)
  heatmap_flux.html             page autonome, fond satellite / sombre / clair (MapLibre), valeur en lux au survol, deux vues :
                                dégradé lissé, ou un point par mesure et par luxmètre (L1, L2, L3, sans lissage) ;
                                case « Lampadaires estimés » (lampadaires_estimes.geojson du rapport) ;
                                image intégrée, connexion requise pour le fond de carte
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import struct
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage

import hexagones as H

# Classes et palette du dashboard (app.py : PALIERS, STOPS), recopiées pour ne pas importer l'appli Dash
PALIERS = [0, 2, 5, 7.5, 10, 15, 20, 30, math.inf]
LUX_MAX = 50
STOPS = [(0, (15, 59, 76)), (5, (30, 95, 105)), (10, (52, 135, 120)), (15, (140, 200, 160)),
         (22, (232, 236, 160)), (30, (247, 214, 120)), (40, (246, 170, 95)), (50, (238, 120, 70))]
TO84 = Transformer.from_crs(2154, 4326, always_xy=True)
# couleurs des types de lampe (app.py : COUL_TYPES, comme carte.html)
COUL_TYPES = {"sodium": "#ff8a2a", "blanc_2700": "#ffd23f", "led_3000": "#3b9bff", "blanc_neutre": "#2ad4b0",
              "blanc_froid": "#b18cff", "inconnu": "#a0a8b5"}
NOMS_TYPES = {"sodium": "Sodium", "blanc_2700": "Blanc 2 700 K", "led_3000": "LED 3 000 K",
              "blanc_neutre": "Blanc neutre", "blanc_froid": "Blanc froid", "inconnu": "Indéterminé"}


def couleur_lux(v):
    v = max(0.0, min(LUX_MAX, float(v)))
    for (a, ca), (b, cb) in zip(STOPS[:-1], STOPS[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return "rgb({},{},{})".format(*(round(x + (y - x) * t) for x, y in zip(ca, cb)))
    return "rgb{}".format(STOPS[-1][1])


COUL_PALIERS = [couleur_lux(m) for m in (1, 3.5, 6.25, 8.75, 12.5, 17.5, 25, 40)]   # milieu de classe


# ------------------------------------------------------------------ DONNÉES
def lire_mesures(dossier: Path, brut: bool = False) -> pd.DataFrame:
    """mesures_pas2m.csv (séparateur ';', virgule décimale), colonnes renommées comme dans app.py."""
    f = dossier / "mesures_pas2m.csv"
    m = pd.read_csv(f, sep=";", decimal=",")
    if m.shape[1] == 1:
        m = pd.read_csv(f)
    m = m.rename(columns={"E": "lux", "E_brut": "lux_brut"})
    if "x" not in m:
        m["x"], m["y"] = Transformer.from_crs(4326, 2154, always_xy=True).transform(m["lon"].to_numpy(),
                                                                                   m["lat"].to_numpy())
    if brut:   # comme _charger_brut : le brut (médiane des 3) placé aux 3 positions de luxmètre
        if "lux_brut" not in m:
            raise SystemExit("Pas de colonne E_brut dans ce rapport.")
        m = m.assign(lux=m["lux_brut"], L1=m["lux_brut"], L2=m["lux_brut"], L3=m["lux_brut"])
    return m


def cellules(m: pd.DataFrame) -> pd.DataFrame:
    """Cellules de 2 m avec E_moy = moyenne des deux sens, et leur centre en Lambert 93."""
    return centres(H.calculer(m)[1])


def centres(p: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le centre en Lambert 93 aux cellules de hexagones.calculer (déjà calculées par le dashboard)."""
    R = H.TAILLE_M / math.sqrt(3)
    p = p.assign(x=R * math.sqrt(3) * (p["q"] + p["r"] / 2), y=R * 1.5 * p["r"])
    return p.dropna(subset=["E_moy"])


def surface(p: pd.DataFrame, res: float, sigma: float, rayon: float):
    """Grille lissée de E_moy (NaN hors du rayon des mesures). Retourne (E, x0, y0) : coin sud-ouest en L93."""
    marge = rayon + 3 * sigma
    x0, y0 = p["x"].min() - marge, p["y"].min() - marge
    nx = int(math.ceil((p["x"].max() + marge - x0) / res)) + 1
    ny = int(math.ceil((p["y"].max() + marge - y0) / res)) + 1
    j = ((p["x"] - x0) / res).astype(int).to_numpy()
    i = ((p["y"] - y0) / res).astype(int).to_numpy()
    W = np.zeros((ny, nx))
    V = np.zeros((ny, nx))
    np.add.at(W, (i, j), 1.0)
    np.add.at(V, (i, j), p["E_moy"].to_numpy(float))
    # lissage normalisé : moyenne pondérée des cellules voisines, sans tirer les bords vers 0
    Gw = ndimage.gaussian_filter(W, sigma / res, mode="constant")
    Gv = ndimage.gaussian_filter(V, sigma / res, mode="constant")
    E = np.full_like(Gv, np.nan)
    np.divide(Gv, Gw, out=E, where=Gw > 1e-9)
    E[ndimage.distance_transform_edt(W == 0) * res > rayon] = np.nan
    return E, x0, y0


def points_capteurs(m: pd.DataFrame):
    """Un point par mesure et par luxmètre, sans lissage ni moyenne : [lon, lat, lux, capteur, sens, heure].
    Position : même décalage latéral par rapport au cap que dans hexagones.py."""
    cap = np.radians(m["cap"].to_numpy(float))
    gx, gy = -np.cos(cap), np.sin(cap)                   # vecteur « gauche » (L93)
    sens = m["sens"].astype(int).tolist()
    heure = m["heure_locale"].round(4).tolist()
    lignes = []
    for n, (col, dl) in enumerate(H.LATERAL_M.items(), start=1):
        lon, lat = TO84.transform(m["x"].to_numpy() + dl * gx, m["y"].to_numpy() + dl * gy)
        lignes += [list(t) for t in zip(np.round(lon, 7).tolist(), np.round(lat, 7).tolist(),
                                        m[col].round(1).tolist(), [n] * len(m), sens, heure) if not pd.isna(t[2])]
    return lignes


def lire_lampadaires(dossier: Path) -> dict:
    """Lampadaires déjà estimés par getys_rapport.py (aucun recalcul). Collection vide si le fichier manque."""
    f = dossier / "lampadaires_estimes.geojson"
    if not f.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(f.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ RASTER -> POLYGONES
def _aretes(B: np.ndarray) -> np.ndarray:
    """Arêtes du bord des zones vraies de B, orientées zone à gauche, en coins de cellule : (j0, i0, j1, i1)."""
    P = np.pad(B, 1)
    c = P[1:-1, 1:-1]
    morceaux = []
    for vide, (a, b, c_, d) in (
            (P[:-2, 1:-1], (0, 0, 1, 0)),     # bas    : (j, i)     -> (j+1, i)
            (P[1:-1, 2:], (1, 0, 1, 1)),      # droite : (j+1, i)   -> (j+1, i+1)
            (P[2:, 1:-1], (1, 1, 0, 1)),      # haut   : (j+1, i+1) -> (j, i+1)
            (P[1:-1, :-2], (0, 1, 0, 0))):    # gauche : (j, i+1)   -> (j, i)
        i, j = np.nonzero(c & ~vide)
        morceaux.append(np.c_[j + a, i + b, j + c_, i + d, i, j])
    return np.concatenate(morceaux)


def _anneaux(aretes) -> list[list[tuple[int, int]]]:
    """Chaîne les arêtes en anneaux fermés. À un coin partagé en diagonale, on tourne à gauche :
    deux cellules qui ne se touchent que par un coin restent séparées (4-connexité, comme ndimage.label)."""
    depuis = defaultdict(list)
    for k, (sx, sy, _, _) in enumerate(aretes):
        depuis[(sx, sy)].append(k)
    utilise = [False] * len(aretes)
    anneaux = []
    for k0 in range(len(aretes)):
        if utilise[k0]:
            continue
        sx, sy, ex, ey = aretes[k0]
        anneau, k = [(sx, sy)], k0
        while True:
            utilise[k] = True
            sx, sy, ex, ey = aretes[k]
            dx, dy = ex - sx, ey - sy
            cands = [c for c in depuis[(ex, ey)] if not utilise[c] or c == k0]
            if len(cands) > 1:   # priorité : gauche, tout droit, droite
                rang = {(-dy, dx): 0, (dx, dy): 1, (dy, -dx): 2}
                cands.sort(key=lambda c: rang.get((aretes[c][2] - ex, aretes[c][3] - ey), 3))
            k = cands[0]
            if k == k0:
                break
            anneau.append((ex, ey))
        anneaux.append(_sans_alignes(anneau))
    return anneaux


def _sans_alignes(pts):
    """Retire les sommets au milieu d'un segment droit, puis ferme l'anneau."""
    n = len(pts)
    garde = []
    for t in range(n):
        (ax, ay), (bx, by), (cx, cy) = pts[t - 1], pts[t], pts[(t + 1) % n]
        if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) != 0:
            garde.append(pts[t])
    return garde + garde[:1]


def _aire(anneau):
    a = np.asarray(anneau, float)
    return 0.5 * np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1])


def polygoniser(B: np.ndarray):
    """Polygones des zones 4-connexes de B : [[extérieur (anti-horaire), trous (horaires)...], ...]."""
    lab, _ = ndimage.label(B)
    ar = _aretes(B)
    if not len(ar):
        return []
    ar = ar[np.argsort(lab[ar[:, 4], ar[:, 5]], kind="stable")]
    zones = lab[ar[:, 4], ar[:, 5]]
    coupures = np.flatnonzero(np.diff(zones)) + 1
    polys = []
    for groupe in np.split(ar[:, :4], coupures):
        anneaux = _anneaux([tuple(r) for r in groupe.tolist()])
        ext = [a for a in anneaux if _aire(a) > 0]
        trous = [a for a in anneaux if _aire(a) < 0]
        polys.append(ext[:1] + trous)          # une zone 4-connexe a un seul contour extérieur
    return polys


def isobandes(E, x0, y0, res):
    """GeoJSON : un MultiPolygon par classe d'éclairement."""
    ok = ~np.isnan(E)
    classe = np.digitize(np.where(ok, E, 0), PALIERS[1:-1])
    feats = []
    for k in range(len(PALIERS) - 1):
        B = ok & (classe == k)
        if not B.any():
            continue
        polys = polygoniser(B)
        coords = []
        for poly in polys:
            anneaux = []
            for an in poly:
                a = np.asarray(an, float)
                lon, lat = TO84.transform(x0 + a[:, 0] * res, y0 + a[:, 1] * res)
                anneaux.append(np.c_[np.round(lon, 7), np.round(lat, 7)].tolist())
            coords.append(anneaux)
        haut = PALIERS[k + 1]
        feats.append({"type": "Feature", "properties": {
            "classe": f"{PALIERS[k]:g}–{haut:g} lx" if haut < math.inf else f"> {PALIERS[k]:g} lx",
            "E_min": PALIERS[k], "E_max": haut if haut < math.inf else None,
            "couleur": COUL_PALIERS[k], "surface_m2": round(float(B.sum()) * res * res),
            "n_zones": len(polys)},
            "geometry": {"type": "MultiPolygon", "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


def points(p: pd.DataFrame):
    """GeoJSON des centres de cellules de 2 m (données d'entrée de la heatmap)."""
    def val(v):
        return None if pd.isna(v) else round(float(v), 1)
    cols = ["_lon", "_lat", "E_moy", "E_sens1", "E_sens2", "n_sens", "n_passages"]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
         "properties": {"E_moy": val(e), "E_sens1": val(e1), "E_sens2": val(e2),
                        "n_sens": int(ns), "n_passages": int(npa)}}
        for lon, lat, e, e1, e2, ns, npa in zip(*(p[c].tolist() for c in cols))]}


# ------------------------------------------------------------------ IMAGE EN DÉGRADÉ
L93_VERS_3857 = Transformer.from_crs(2154, 3857, always_xy=True)
W3857_VERS_L93 = Transformer.from_crs(3857, 2154, always_xy=True)
W3857_VERS_84 = Transformer.from_crs(3857, 4326, always_xy=True)
IMAGE_MAX_PX = 8192          # taille de texture sûre pour WebGL
LUX_PAR_CODE = 0.2           # grille de survol : valeur codée sur un octet, pas de 0,2 lx (255 = pas de mesure)


def image_degrade(E, x0, y0, res, pixel_m=1.0):
    """Surface lissée (grille L93, NaN hors mesures) -> image RGBA en Web Mercator, prête pour la carte.
    Couleur : rampe continue STOPS (0 à 50 lx), interpolation bilinéaire. Transparence : masque des mesures
    adouci, le bord des zones mesurées s'estompe sur ~1 m.
    Retourne (rgba, codes de survol, cadre : coins lon/lat + origine, pas et taille en Web Mercator)."""
    ok = ~np.isnan(E)
    ii, jj = np.nonzero(ok)
    bx = x0 + np.array([jj.min(), jj.max() + 1]) * res
    by = y0 + np.array([ii.min(), ii.max() + 1]) * res
    mx, my = L93_VERS_3857.transform(*[g.ravel() for g in np.meshgrid(bx, by)])
    X0, X1, Y0, Y1 = mx.min(), mx.max(), my.min(), my.max()
    lat = W3857_VERS_84.transform((X0 + X1) / 2, (Y0 + Y1) / 2)[1]
    pas = pixel_m / math.cos(math.radians(lat))          # 1 m au sol = 1/cos(lat) m en Web Mercator
    pas *= max(1.0, max(X1 - X0, Y1 - Y0) / pas / IMAGE_MAX_PX)
    w, h = int(math.ceil((X1 - X0) / pas)), int(math.ceil((Y1 - Y0) / pas))
    X1, Y0 = X0 + w * pas, Y1 - h * pas

    # valeurs prolongées hors mesures (plus proche voisin) : pas de liseré sombre au bord du dégradé
    idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    Ef = E[tuple(idx)]
    A = ndimage.gaussian_filter(ok.astype(float), 0.75, mode="constant")
    px = X0 + (np.arange(w) + 0.5) * pas
    v = np.empty((h, w), np.float32)
    a = np.empty((h, w), np.float32)
    for r0 in range(0, h, 256):                          # par bandes de lignes : mémoire bornée
        py = Y1 - (np.arange(r0, min(h, r0 + 256)) + 0.5) * pas
        gx, gy = W3857_VERS_L93.transform(*[g.ravel() for g in np.meshgrid(px, py)])
        fi, fj = (gy - y0) / res - 0.5, (gx - x0) / res - 0.5
        v[r0:r0 + len(py)] = ndimage.map_coordinates(Ef, [fi, fj], order=1, mode="nearest").reshape(len(py), w)
        a[r0:r0 + len(py)] = ndimage.map_coordinates(A, [fi, fj], order=1, cval=0.0).reshape(len(py), w)
    t = np.clip((a - 0.15) / 0.45, 0, 1)
    alpha = t * t * (3 - 2 * t)                          # smoothstep

    vs, cs = [s for s, _ in STOPS], np.array([c for _, c in STOPS], float)
    vc = np.clip(v, 0, LUX_MAX)
    rgba = np.empty((h, w, 4), np.uint8)
    for k in range(3):
        rgba[..., k] = np.round(np.interp(vc, vs, cs[:, k]))
    rgba[..., 3] = np.round(alpha * 255)
    codes = np.where(alpha > 0.5, np.minimum(np.round(v / LUX_PAR_CODE), 254), 255).astype(np.uint8)
    coins = [list(W3857_VERS_84.transform(x, y)) for x, y in ((X0, Y1), (X1, Y1), (X1, Y0), (X0, Y0))]
    return rgba, codes, {"coins": coins, "x0": X0, "y1": Y1, "pas": pas, "w": w, "h": h}


def png(rgba) -> bytes:
    """PNG RGBA 8 bits, sans dépendance."""
    h, w, _ = rgba.shape
    brut = np.concatenate([np.zeros((h, 1), np.uint8), rgba.reshape(h, -1)], axis=1).tobytes()

    def bloc(typ, data):
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + bloc(b"IDAT", zlib.compress(brut, 6)) + bloc(b"IEND", b""))


# ------------------------------------------------------------------ VISIONNEUSE
PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Éclairement — __TITRE__</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<style>
 html,body,#carte{margin:0;height:100%;background:#0b0f14;font:13px/1.45 Inter,system-ui,-apple-system,sans-serif}
 #panneau{position:absolute;top:14px;left:14px;width:280px;padding:14px 16px;color:#e6edf3;
   background:rgba(13,17,23,.82);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.08);border-radius:12px;
   box-shadow:0 8px 28px rgba(0,0,0,.35)}
 h1{font-size:15px;font-weight:600;margin:0} .sous{color:#9da7b3;font-size:12px;margin:3px 0 12px}
 .rampe{height:12px;border-radius:6px;background:linear-gradient(90deg,__DEGRADE__)}
 .graduations{display:flex;justify-content:space-between;color:#9da7b3;font-size:11px;margin-top:4px;
   font-variant-numeric:tabular-nums}
 .ligne{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:12px;color:#c9d1d9}
 input[type=range]{width:130px;accent-color:#f6aa5f} .seg{display:flex;background:#0b0f14;border-radius:8px;padding:2px}
 .seg button{border:0;background:none;color:#9da7b3;font:inherit;padding:3px 10px;border-radius:6px;cursor:pointer}
 .seg button.on{background:#30363d;color:#e6edf3}
 .note{color:#9da7b3;font-size:11.5px;margin-top:6px}
 label.ligne{cursor:pointer} input[type=checkbox]{width:16px;height:16px;accent-color:#f6aa5f}
 #legende-lampes{display:grid;grid-template-columns:1fr 1fr;gap:3px 8px;margin-top:8px;font-size:11.5px;color:#c9d1d9}
 #legende-lampes[hidden]{display:none} #legende-lampes i{display:inline-block;width:8px;height:8px;border-radius:50%;
   border:1.5px solid #fff;margin-right:5px}
 .export{margin-top:10px;display:flex;align-items:center;gap:6px;flex-wrap:wrap;color:#9da7b3;font-size:11.5px}
 .export a{color:#e6edf3;text-decoration:none;border:1px solid rgba(255,255,255,.16);border-radius:6px;padding:3px 8px;
   background:rgba(255,255,255,.04)} .export a:hover{border-color:#6cc08b;color:#fff}
 #valeur{position:absolute;pointer-events:none;padding:5px 9px;border-radius:7px;background:rgba(13,17,23,.9);
   color:#e6edf3;font-size:12px;border:1px solid rgba(255,255,255,.1);transform:translate(12px,-50%);display:none;
   white-space:nowrap} #valeur b{font-size:14px}
</style></head><body><div id="carte"></div>
<div id="panneau"><h1>Éclairement</h1><div class="sous">__TITRE__<br>__PARAMS__</div>
 <div class="rampe"></div><div class="graduations"><span>0</span><span>10</span><span>20</span><span>30</span>
 <span>40</span><span>50+ lx</span></div>
 <div class="export">Exporter en GeoJSON :
  <a download href="__EXPORT_ZONES__" title="Surface lissée : un MultiPolygon par classe de lux">Zones par classe</a>
  <a download href="__EXPORT_CELLULES__" title="Centres des cellules de 2 m : moyenne et valeur de chaque sens">Cellules de 2 m</a></div>
 <div class="ligne">Vue <div class="seg" id="vue"><button class="on" data-vue="lisse">Lissée</button>
  <button data-vue="capteurs">3 capteurs</button></div></div>
 <div class="note" id="note">__NOTE_LISSE__</div>
 <label class="ligne" id="ligne-lampes">Lampadaires estimés <input type="checkbox" id="lampes"></label>
 <div id="legende-lampes" hidden></div>
 <div class="ligne">Opacité <input type="range" id="opacite" min="0" max="100" value="90"></div>
 <div class="ligne" id="ligne-fond">Fond <div class="seg" id="fond"><button class="on" data-fond="satellite">Satellite</button>
  <button data-fond="sombre">Sombre</button><button data-fond="clair">Clair</button></div></div></div>
<div id="valeur"></div>
<script>
const CAPTEURS = __CAPTEURS__;
const LAMPES = __LAMPES__, COUL_TYPES = __COUL_TYPES__, NOMS_TYPES = __NOMS_TYPES__;
const IMG = "data:image/png;base64,__IMAGE__", COINS = __COINS__, GRILLE = __GRILLE__;
const ESRI = n => `https://server.arcgisonline.com/ArcGIS/rest/services/${n}/MapServer/tile/{z}/{y}/{x}`;
const carte = new maplibregl.Map({container: "carte", maxZoom: 21, attributionControl: {compact: true},
  bounds: [COINS[3], COINS[1]], fitBoundsOptions: {padding: 40},
  style: {version: 8, sources: {
      satellite: {type: "raster", tiles: [ESRI("World_Imagery")], tileSize: 256, maxzoom: 19,
                  attribution: "Imagerie © Esri, Maxar, Earthstar Geographics"},
      noms: {type: "raster", tiles: [ESRI("Reference/World_Boundaries_and_Places")], tileSize: 256, maxzoom: 19},
      clair: {type: "raster", tiles: [ESRI("Canvas/World_Light_Gray_Base")], tileSize: 256,
              attribution: "Esri, HERE, Garmin, © contributeurs OpenStreetMap", maxzoom: 16},
      sombre: {type: "raster", tiles: [ESRI("Canvas/World_Dark_Gray_Base")], tileSize: 256,
               attribution: "Esri, HERE, Garmin, © contributeurs OpenStreetMap", maxzoom: 16},
      eclairement: {type: "image", url: IMG, coordinates: COINS}},
    layers: [
      {id: "satellite", type: "raster", source: "satellite", paint: {"raster-saturation": -0.35, "raster-brightness-max": 0.7}},
      {id: "sombre", type: "raster", source: "sombre", layout: {visibility: "none"}},
      {id: "clair", type: "raster", source: "clair", layout: {visibility: "none"}},
      {id: "eclairement", type: "raster", source: "eclairement",
       paint: {"raster-resampling": "linear", "raster-opacity": 0.9, "raster-fade-duration": 0}},
      {id: "noms", type: "raster", source: "noms"}]}});
carte.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
carte.addControl(new maplibregl.ScaleControl({unit: "metric"}), "bottom-right");

let vue = "lisse";
const NOTES = {lisse: "__NOTE_LISSE__",
  capteurs: "Valeur de chaque capteur à chaque mesure, sans lissage ni moyenne des deux sens. Zoomer pour les distinguer."};
function appliquerOpacite() {
  const o = document.getElementById("opacite").value / 100;
  carte.setPaintProperty("eclairement", "raster-opacity", o);
  if (carte.getLayer("capteurs")) carte.setPaintProperty("capteurs", "circle-opacity", o);
}
function creerCapteurs() {                     // créée au premier affichage seulement (≈ 100 000 points)
  carte.addSource("capteurs", {type: "geojson", maxzoom: 19, data: {type: "FeatureCollection",
    features: CAPTEURS.map(([lon, lat, lux, capteur, sens, heure]) => ({type: "Feature",
      geometry: {type: "Point", coordinates: [lon, lat]}, properties: {lux, capteur, sens, heure}}))}});
  carte.addLayer({id: "capteurs", type: "circle", source: "capteurs", paint: {
    "circle-color": ["interpolate", ["linear"], ["get", "lux"], __STOPS__],
    // rayon ≈ 0,22 m au sol à fort zoom (les 3 capteurs restent distincts), 1 px minimum de loin
    "circle-radius": ["interpolate", ["exponential", 2], ["zoom"], 14, 1, 18, 1.4, 22, 18]}}, "noms");
}
document.getElementById("opacite").oninput = appliquerOpacite;
const caseLampes = document.getElementById("lampes"), legende = document.getElementById("legende-lampes");
if (!LAMPES.features.length) document.getElementById("ligne-lampes").style.display = "none";
const compte = {};
LAMPES.features.forEach(f => { compte[f.properties.type] = (compte[f.properties.type] || 0) + 1; });
legende.innerHTML = Object.keys(NOMS_TYPES).filter(t => compte[t])
  .map(t => `<span><i style="background:${COUL_TYPES[t]}"></i>${NOMS_TYPES[t]} (${compte[t]})</span>`).join("");
function basculerLampes() {
  const oui = caseLampes.checked;
  legende.hidden = !oui;
  if (!carte.getLayer("lampes")) {
    if (!oui) return;
    carte.addSource("lampes", {type: "geojson", data: LAMPES});
    const rayon = k => ["interpolate", ["linear"], ["zoom"], 13, 2 * k, 17, 4.5 * k, 21, 9 * k];
    carte.addLayer({id: "lampes-halo", type: "circle", source: "lampes",   // liseré sombre : lisible sur le dégradé
      paint: {"circle-radius": rayon(1.6), "circle-color": "rgba(5,8,12,0.75)"}});
    carte.addLayer({id: "lampes", type: "circle", source: "lampes", paint: {"circle-radius": rayon(1),
      "circle-color": ["match", ["get", "type"], ...Object.entries(COUL_TYPES).flat(), "#a0a8b5"],
      "circle-stroke-color": "#fff", "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 13, 0.5, 18, 2]}});
  }
  for (const c of ["lampes-halo", "lampes"]) carte.setLayoutProperty(c, "visibility", oui ? "visible" : "none");
  bulle.style.display = "none";
}
caseLampes.onchange = () => carte.getLayer("eclairement") ? basculerLampes() : carte.once("load", basculerLampes);
document.querySelectorAll("#vue button").forEach(b => b.onclick = () => {
  if (!carte.isStyleLoaded() && !carte.getLayer("eclairement")) return;
  document.querySelectorAll("#vue button").forEach(x => x.classList.toggle("on", x === b));
  vue = b.dataset.vue;
  if (vue === "capteurs" && !carte.getLayer("capteurs")) creerCapteurs();
  carte.setLayoutProperty("eclairement", "visibility", vue === "lisse" ? "visible" : "none");
  if (carte.getLayer("capteurs"))
    carte.setLayoutProperty("capteurs", "visibility", vue === "capteurs" ? "visible" : "none");
  appliquerOpacite();
  document.getElementById("note").textContent = NOTES[vue];
  bulle.style.display = "none";
});
function choisirFond(fond) {
  if (!["satellite", "sombre", "clair"].includes(fond)) return;
  document.querySelectorAll("#fond button").forEach(x => x.classList.toggle("on", x.dataset.fond === fond));
  const appliquer = () => {
    for (const c of ["satellite", "sombre", "clair"])
      carte.setLayoutProperty(c, "visibility", c === fond ? "visible" : "none");
    carte.setLayoutProperty("noms", "visibility", fond === "satellite" ? "visible" : "none");
  };
  carte.isStyleLoaded() ? appliquer() : carte.once("load", appliquer);
}
document.querySelectorAll("#fond button").forEach(b => b.onclick = () => choisirFond(b.dataset.fond));
// intégrée au dashboard : le fond suit celui du dashboard (#fond=…, changé sans recharger la page)
const fondAdresse = () => (location.hash.match(/fond=(\\w+)/) || [])[1];
if (/integre=1/.test(location.hash)) document.getElementById("ligne-fond").style.display = "none";
if (fondAdresse()) choisirFond(fondAdresse());
window.addEventListener("hashchange", () => choisirFond(fondAdresse()));

// valeur au survol : grille d'octets (0,2 lx par pas) alignée sur les pixels de l'image, compressée (zlib)
let codes = null;
fetch("data:application/octet-stream;base64," + GRILLE.donnees).then(r => r.body.pipeThrough(new DecompressionStream("deflate")))
  .then(s => new Response(s).arrayBuffer()).then(b => { codes = new Uint8Array(b); });
const bulle = document.getElementById("valeur"), R = 6378137;
const nombre = v => v.toLocaleString("fr-FR", {maximumFractionDigits: 1, minimumFractionDigits: 1});
const NOM_CAPTEUR = {1: "L1 (droite)", 2: "L2 (centre)", 3: "L3 (gauche)"};
function afficher(e, texte) {
  bulle.innerHTML = texte;
  Object.assign(bulle.style, {display: "block", left: e.point.x + "px", top: e.point.y + "px"});
}
const NOM_COTE = {gauche: "gauche", droite: "droite", non_tranche: "non tranché"};
function texteLampe(p) {
  return `<b>${p.type_libelle}</b><br>Puissance estimée : ${Math.round(p.puissance_w)} W `
    + `(${Math.round(p.puissance_min_w)}–${Math.round(p.puissance_max_w)} W)<br>Côté : ${NOM_COTE[p.cote] || p.cote} · `
    + `pic ${nombre(p.E_pic)} lx · ${p.n_passages} passage(s)`;
}
carte.on("mousemove", e => {
  if (caseLampes.checked && carte.getLayer("lampes")) {
    const lampe = carte.queryRenderedFeatures(e.point, {layers: ["lampes"]})[0];
    if (lampe) { afficher(e, texteLampe(lampe.properties)); return; }
  }
  if (vue === "capteurs") {
    const marge = 3, f = carte.queryRenderedFeatures([[e.point.x - marge, e.point.y - marge],
      [e.point.x + marge, e.point.y + marge]], {layers: ["capteurs"]})[0];
    if (!f) { bulle.style.display = "none"; return; }
    const p = f.properties, min = Math.round(p.heure * 60);
    afficher(e, `<b>${nombre(p.lux)} lx</b> · ${NOM_CAPTEUR[p.capteur]}<br>sens ${p.sens} · `
      + `${Math.floor(min / 60) % 24}h${String(min % 60).padStart(2, "0")}`);
    return;
  }
  if (!codes) return;
  const x = R * e.lngLat.lng * Math.PI / 180, y = R * Math.log(Math.tan(Math.PI / 4 + e.lngLat.lat * Math.PI / 360));
  const c = Math.floor((x - GRILLE.x0) / GRILLE.pas), l = Math.floor((GRILLE.y1 - y) / GRILLE.pas);
  const v = c >= 0 && l >= 0 && c < GRILLE.w && l < GRILLE.h ? codes[l * GRILLE.w + c] : 255;
  if (v === 255) { bulle.style.display = "none"; return; }
  afficher(e, `<b>${nombre(v * GRILLE.lux)} lx</b> · moyenne des deux sens`);
});
carte.on("mouseout", () => { bulle.style.display = "none"; });
</script></body></html>
"""


def page(image_png: bytes, codes, cadre, capteurs, lampes, titre, params,
         exports=("heatmap_flux.geojson", "heatmap_flux_points.geojson")):
    """exports : liens (relatifs à la page) des deux GeoJSON proposés dans la légende."""
    grille = {k: cadre[k] for k in ("x0", "y1", "pas", "w", "h")}
    grille.update(lux=LUX_PAR_CODE, donnees=base64.b64encode(zlib.compress(codes.tobytes(), 9)).decode())
    degrade = ",".join(f"rgb{c} {100 * v / LUX_MAX:g}%" for v, c in STOPS)
    stops = ",".join(f'{v},"rgb{c}"' for v, c in STOPS)
    return (PAGE.replace("__STOPS__", stops).replace("__NOTE_LISSE__", "Moyenne des deux sens, surface lissée.")
            .replace("__CAPTEURS__", json.dumps(capteurs, separators=(",", ":")))
            .replace("__LAMPES__", json.dumps(lampes, separators=(",", ":")))
            .replace("__COUL_TYPES__", json.dumps(COUL_TYPES)).replace("__NOMS_TYPES__", json.dumps(NOMS_TYPES))
            .replace("__TITRE__", titre).replace("__PARAMS__", params).replace("__DEGRADE__", degrade)
            .replace("__COINS__", json.dumps(cadre["coins"])).replace("__GRILLE__", json.dumps(grille))
            .replace("__EXPORT_ZONES__", exports[0]).replace("__EXPORT_CELLULES__", exports[1])
            .replace("__IMAGE__", base64.b64encode(image_png).decode()))


RES_M, SIGMA_M, RAYON_M, IMAGE_M = 2.0, 4.0, 6.0, 1.0      # réglages par défaut (script et dashboard)


def page_heatmap(p, m_capteurs, lampes, titre, brut=False, res=RES_M, sigma=SIGMA_M, rayon=RAYON_M, image_m=IMAGE_M):
    """Page heatmap_flux.html : la même pour le script et pour l'onglet Heatmap du dashboard.
    p : cellules avec centres (cellules / centres) ; m_capteurs : mesures pour la vue 3 capteurs.
    Retourne (html, png, cadre, (E, x0, y0))."""
    E, x0, y0 = surface(p, res, sigma, rayon)
    rgba, codes, cadre = image_degrade(E, x0, y0, res, image_m)
    image_png = png(rgba)
    params = (f"Grille {res:g} m · lissage σ = {sigma:g} m · ≤ {rayon:g} m des mesures"
              + (" · signal brut" if brut else ""))
    return (page(image_png, codes, cadre, points_capteurs(m_capteurs), lampes, titre, params,
                 tuple(f"{n}{'_brut' if brut else ''}.geojson" for n in ("heatmap_flux", "heatmap_flux_points"))),
            image_png, cadre, (E, x0, y0))


# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier", type=Path, help="dossier d'un rapport (contient mesures_pas2m.csv)")
    ap.add_argument("--res-m", type=float, default=RES_M, help="pas de la grille lissée (m)")
    ap.add_argument("--sigma-m", type=float, default=SIGMA_M, help="écart-type du lissage gaussien (m)")
    ap.add_argument("--rayon-m", type=float, default=RAYON_M, help="distance max. à une cellule mesurée (m)")
    ap.add_argument("--image-m", type=float, default=IMAGE_M, help="taille d'un pixel de l'image en dégradé (m au sol)")
    ap.add_argument("--brut", action="store_true", help="signal avant lissage du bruit (E_brut)")
    ap.add_argument("-o", "--sortie", type=Path, help="dossier de sortie (défaut : celui du rapport)")
    a = ap.parse_args()

    t0 = time.time()
    sortie = a.sortie or a.dossier
    sortie.mkdir(parents=True, exist_ok=True)
    m = lire_mesures(a.dossier, a.brut)
    p = cellules(m)
    html, image_png, cadre, (E, x0, y0) = page_heatmap(
        p, lire_mesures(a.dossier), lire_lampadaires(a.dossier), a.dossier.name, a.brut,
        a.res_m, a.sigma_m, a.rayon_m, a.image_m)
    bandes = isobandes(E, x0, y0, a.res_m)
    pts = points(p)
    suffixe = "_brut" if a.brut else ""
    (sortie / f"heatmap_flux{suffixe}.geojson").write_text(json.dumps(bandes, separators=(",", ":")), encoding="utf-8")
    (sortie / f"heatmap_flux_points{suffixe}.geojson").write_text(json.dumps(pts, separators=(",", ":")),
                                                                 encoding="utf-8")
    (sortie / f"heatmap_flux{suffixe}.png").write_bytes(image_png)
    (sortie / f"heatmap_flux{suffixe}.html").write_text(html, encoding="utf-8")
    print(f"{len(p)} cellules de 2 m, grille {E.shape[1]}×{E.shape[0]}, {len(bandes['features'])} classes, "
          f"{sum(f['properties']['n_zones'] for f in bandes['features'])} zones, image {cadre['w']}×{cadre['h']} px"
          f" ({len(image_png) / 1e6:.1f} Mo) — {time.time() - t0:.1f} s")
    for f in bandes["features"]:
        pr = f["properties"]
        print(f"  {pr['classe']:>10} : {pr['surface_m2']:>8} m², {pr['n_zones']} zone(s)")
    print(f"-> {sortie / f'heatmap_flux{suffixe}.html'}")


if __name__ == "__main__":
    main()
