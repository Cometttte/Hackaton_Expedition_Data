"""Tableau de bord éclairage public — upload TDMS -> serveur -> affichage des sorties.

Lectures des mesures (cf. cahier des charges GETYS) :
  * Moyenne au tronçon : filaire OSM, une moyenne par sens de circulation puis moyenne des deux sens
  * Hexagones de 2 m   : bande continue le long du trajet (3 luxmètres décalés latéralement, comme
    getys_spatialisation.py), multi-passages, même règle par sens
  * Heatmap : la page de heatmap_flux.py (dégradé lissé ou 3 capteurs, lampadaires estimés, export GeoJSON),
    intégrée telle quelle ; c'est aussi la page du bouton « Carte satellite » (plein écran, fond satellite)
Carte : Leaflet (dash-leaflet), comme carte.html. Les données de chaque calque sont servies à part (gzip, cache
navigateur) ; couleurs et seuils SDAL sont appliqués dans le navigateur (assets/carte_leaflet.js).
La conformité (SDAL : un seuil d'éclairement moyen par type de voie) ne s'applique qu'aux tronçons.
Conformité SDAL : vue Tronçons. Comparaison avant / après lissage du bruit (colonne E_brut de mesures_pas2m.csv)
et valeurs aberrantes retirées : vues Tronçons et Hexagones. Lampadaires estimés : vue Heatmap seulement.
Le fond vaut pour toutes les vues ; thème clair / sombre (bouton de l'en-tête, mémorisé dans le navigateur).
"""
import base64
import gzip
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
from dash import ALL, ClientsideFunction, Input, Output, Patch, State, ctx, dash_table, dcc, html  # noqa: F401
from flask import Response, abort, request, send_from_directory
from pyproj import Transformer
from scipy.spatial import cKDTree

import doc as DOC
import heatmap_flux as HF
import hexagones as H
import operateur as OP
import rapport_affaires as RA
import sdal_fichier as SF
import troncons as T

# ------------------------------------------------------------------ CONFIG
ICI = Path(__file__).resolve().parent
RACINE = ICI.parent                           # racine du dépôt (getys_rapport.py, rapports/)
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:5001/upload")   # serveur d'analyse (serveur.py)
RAPPORTS_DIR = Path(os.environ.get("RAPPORTS_DIR", RACINE / "rapports"))     # sorties de getys_rapport.py
LIEN_HACKATHON = "https://www.bouygues-es.fr"                                # Hackathon 2026 Bouygues E&S
# Fonds de carte (tuiles raster, gratuites, sans clé). Satellite : mêmes tuiles que carte.html (getys_rapport.py)
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
ATTR_ESRI = "© Esri, HERE, Garmin, © OpenStreetMap contributors"
ZOOM_MAX = 21            # fonds vectoriels (OpenFreeMap) : nets à tous les zooms
# Clair / Sombre : fond vectoriel OpenFreeMap dessiné par MapLibre dans Leaflet (assets/carte_leaflet.js,
# getys.appliquerFond) : aucune tuile raster ici. Satellite : imagerie Esri, zoom natif 19.
# Chaque calque raster : (url, attribution, zoom natif max, zoom min, zoom max, classe CSS).
FONDS = {
    "sombre": [],
    "clair": [],
    "satellite": [(ESRI.format("World_Imagery"), "Imagerie © Esri, Maxar, Earthstar Geographics", 19, 0, ZOOM_MAX, ""),
                  (ESRI.format("Reference/World_Boundaries_and_Places"), "", 19, 0, ZOOM_MAX, "")],
}


def tuiles(fond, prefixe="tuiles"):
    """Fonds nets à tous les zooms : chaque source n'est affichée que là où elle a de vraies tuiles."""
    # id propre à chaque fond : le calque est recréé (sinon Leaflet garde la tuile OSM sans changer son filtre).
    # Haute résolution (écrans Retina) : detectRetina charge les tuiles du zoom suivant à demi-taille, donc 4x plus
    # de pixels. Seulement là où ce zoom existe (OSM, satellite : 19) ; le zoom natif est alors abaissé d'un cran
    # pour que la tuile demandée (zoom + 1) reste disponible. Pas sur les fonds gris Esri (zoom natif 16).
    calques = []
    for i, (u, a, nz, zmin, zmax, cl) in enumerate(FONDS[fond]):
        retina = nz >= 19
        calques.append(dl.TileLayer(id=f"{prefixe}-{fond}-{i}", url=u, attribution=a, minZoom=zmin, maxZoom=zmax,
                                    maxNativeZoom=nz - 1 if retina else nz, detectRetina=retina, className=cl))
    return calques


# Charte GETYS (couleurs relevées sur le logo de Getys_Hackathon.pdf)
VERT_FONCE, VERT_GETYS, LIME = "#1F6E43", "#3E9C61", "#8DBF2E"
VERT, ROUGE, ORANGE = "#2E9E5B", "#E75113", "#F2A900"   # couleurs d'état (graphiques, KPI)
# carte en mode « Conformité SDAL » : sous le seuil en bleu (bleu clair : un seul sens mesuré), conforme jusqu'à
# 40 lx en vert, plus de 40 lx en rouge
AMBRE = VERT_GETYS                                         # couleur d'accent des graphiques
CYAN = "#00B5E2"                                           # sélection sur la carte
CARTE_H = 640
LUX_MAX = 50                                  # haut de l'échelle des lux (comme carte.html)

# Palette froid -> chaud de carte.html (getys_rapport.py), de 0 à 50 lx
STOPS = [(0, (15, 59, 76)), (5, (30, 95, 105)), (10, (52, 135, 120)), (15, (140, 200, 160)),
         (22, (232, 236, 160)), (30, (247, 214, 120)), (40, (246, 170, 95)), (50, (238, 120, 70))]


def couleur_lux(v):
    """Même interpolation que rgb() dans carte_template.html."""
    v = max(0.0, min(LUX_MAX, float(v)))
    for (a, ca), (b, cb) in zip(STOPS[:-1], STOPS[1:]):
        if v <= b:
            t = (v - a) / (b - a)
            return "rgb({},{},{})".format(*(round(x + (y - x) * t) for x, y in zip(ca, cb)))
    return "rgb{}".format(STOPS[-1][1])


# classes d'éclairement pour colorer les tronçons (lx) : couleur au milieu de chaque classe
PALIERS = [0, 2, 5, 7.5, 10, 15, 20, 30, math.inf]
COUL_PALIERS = [couleur_lux(m) for m in (1, 3.5, 6.25, 8.75, 12.5, 17.5, 25, 40)]
COUL_TYPES = {"sodium": "#ff8a2a", "blanc_2700": "#ffd23f", "led_3000": "#3b9bff", "blanc_neutre": "#2ad4b0",
              "blanc_froid": "#b18cff", "inconnu": "#a0a8b5"}          # couleurs de carte.html
# valeurs aberrantes retirées par le filtrage du bruit, couleur par motif
COUL_MOTIFS = {"trop haut": "#ff3b3b", "trop bref": "#ff9f43", "variation à l'arrêt": "#c77dff"}

TO93 = Transformer.from_crs(4326, 2154, always_xy=True)
TO84 = Transformer.from_crs(2154, 4326, always_xy=True)

# Thème Plotly commun : fonds transparents, police du dashboard
pio.templates["clair"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["clair"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Roboto, Arial, sans-serif", color="#26332B", size=12),
    xaxis=dict(gridcolor="#E3ECE5", zerolinecolor="#DCE7DF", linecolor="#BFCBC2"),
    yaxis=dict(gridcolor="#E3ECE5", zerolinecolor="#DCE7DF", linecolor="#BFCBC2"),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=VERT_GETYS, font_color="#26332B"),
)
pio.templates.default = "clair"
BARRE_COULEUR = dict(thickness=12, x=0.99, xanchor="right", len=0.55, bgcolor="rgba(13,17,23,0.75)",
                     tickfont=dict(color="#e6edf3"), title_font=dict(color="#e6edf3"))

app = dash.Dash(__name__, title="Éclairage public", suppress_callback_exceptions=True,
                external_scripts=[],          # MapLibre est chargé à la demande par assets/carte_leaflet.js
                compress=True,                    # réponses des callbacks compressées (flask-compress)
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,
                                      "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap",
                                      "https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css"])


@app.server.route("/rapport/<path:chemin>")
def servir_rapport(chemin):
    """Sert rapport.html / carte.html produits par getys_rapport.py."""
    racine = RAPPORTS_DIR.resolve()
    if not (racine / chemin).is_file():
        abort(404)
    return send_from_directory(racine, chemin)


COUCHES = ("troncons", "hexagones")


@app.server.route("/geo/<nom>/<variante>/<couche>.geojson")
def servir_couche(nom, variante, couche):
    """Données d'un calque de la carte (GeoJSON compressé), servies à part des callbacks : Leaflet les
    télécharge par URL, le navigateur les garde en cache (l'URL change avec le rapport)."""
    dossiers = {d.name: d for d in lister_rapports()}
    if nom not in dossiers or variante not in ("lisse", "brut") or couche not in COUCHES:
        abort(404)
    d = dossiers[nom]
    corps, etag = _couche_gz(str(d), d.stat().st_mtime, variante == "brut", couche)
    if request.headers.get("If-None-Match") == etag:
        return Response(status=304)
    rep = Response(corps, mimetype="application/geo+json")
    rep.headers.update({"Content-Encoding": "gzip", "ETag": etag, "Vary": "Accept-Encoding",
                        "Cache-Control": "public, max-age=86400, immutable"})
    return rep


def _r(v, d=1):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), d)


def _geojson(data, couche):
    """Propriétés courtes (le fichier part au navigateur) : E = lux, s1/s2 = sens, np = passages…"""
    f = []
    if couche == "troncons" and data["troncons"] is not None:
        for r in data["troncons"].itertuples():
            lon, lat = data["geoms"][r.id]
            f.append({"type": "Feature", "geometry": {"type": "LineString",
                                                      "coordinates": [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]},
                      "properties": {"id": r.id, "n": r.nom, "c": r.classe, "lib": T.CLASSES[r.classe][0],
                                     "sd": T.CLASSES[r.classe][2], "E": _r(r.E_moy), "s1": _r(r.E_sens1),
                                     "s2": _r(r.E_sens2), "n1": int(r.n_pass_sens1), "n2": int(r.n_pass_sens2),
                                     "np": int(r.n_passages), "ns": int(r.n_sens), "su": int(bool(r.sens_unique))}})
    elif couche == "hexagones" and data["hex"] is not None:
        gj, p = data["hex"]
        for k, (feat, r) in enumerate(zip(gj["features"], p.itertuples())):
            f.append({**feat, "properties": {"k": k, "E": _r(r.E_moy), "s1": _r(r.E_sens1),
                                             "s2": _r(r.E_sens2), "np": int(r.n_passages)}})
    return {"type": "FeatureCollection", "features": f}


@lru_cache(maxsize=8)
def _couche_gz(dossier, mtime, brut, couche):
    corps = json.dumps(_geojson(charger(dossier, brut=brut), couche), separators=(",", ":")).encode()
    return gzip.compress(corps, 6), f'"{couche}-{int(mtime)}-{int(brut)}"'


@app.server.route("/heatmap/<nom>/<variante>.html")
def servir_heatmap(nom, variante):
    """Page de heatmap_flux.py pour une tournée, générée au premier appel (quelques secondes) puis en cache."""
    dossiers = {d.name: d for d in lister_rapports()}
    if nom not in dossiers or variante not in ("lisse", "brut"):
        abort(404)
    d = dossiers[nom]
    etag = f'"heatmap-{int(d.stat().st_mtime)}-{variante}"'
    if request.headers.get("If-None-Match") == etag:
        return Response(status=304)
    rep = Response(_page_heatmap(str(d), d.stat().st_mtime, variante == "brut"), mimetype="text/html")
    rep.headers.update({"ETag": etag, "Cache-Control": "public, max-age=86400, immutable"})
    return rep


@lru_cache(maxsize=2)
def _page_heatmap(dossier, _mtime, brut):
    """Mêmes entrées que le script : cellules de 2 m (moyenne des deux sens), mesures filtrées pour les
    3 capteurs, lampadaires_estimes.geojson ; brut : surface recalculée sur E_brut (comme --brut)."""
    data = charger(dossier, brut=brut)
    html_page, *_ = HF.page_heatmap(HF.centres(data["hex"][1]), charger(dossier)["mesures"],
                                     HF.lire_lampadaires(Path(dossier)), Path(dossier).name, brut=data["brut"])
    return html_page.encode("utf-8")


@app.server.route("/operateur/<nom>.json")
def servir_trace_brute(nom):
    """Vue Opérateur : trace du .tdms brut (valeurs des 3 capteurs à chaque position GPS, heure)."""
    f = OP.chemin_tdms(f"{nom}.tdms")
    if f is None:
        abort(404)
    corps = _trace_gz(str(f), f.stat().st_mtime)
    rep = Response(corps, mimetype="application/json")
    rep.headers.update({"Content-Encoding": "gzip", "Vary": "Accept-Encoding", "Cache-Control": "no-cache"})
    return rep


@lru_cache(maxsize=3)
def _trace_gz(chemin, _mtime):
    return gzip.compress(json.dumps(OP.trace(OP.lire_brut(Path(chemin))), separators=(",", ":")).encode(), 6)


@app.server.route("/heatmap/<nom>/<fichier>.geojson")
def servir_heatmap_geojson(nom, fichier):
    """Exports de la légende de la heatmap : mêmes fichiers que heatmap_flux.py (heatmap_flux.geojson :
    zones par classe de lux ; heatmap_flux_points.geojson : cellules de 2 m), calculés à la demande."""
    dossiers = {d.name: d for d in lister_rapports()}
    brut = fichier.endswith("_brut")
    quoi = fichier.removesuffix("_brut")
    if nom not in dossiers or quoi not in ("heatmap_flux", "heatmap_flux_points"):
        abort(404)
    d = dossiers[nom]
    corps = _heatmap_geojson(str(d), d.stat().st_mtime, brut, quoi)
    rep = Response(corps, mimetype="application/geo+json")
    rep.headers["Content-Disposition"] = f'attachment; filename="{fichier}_{quote(nom)}.geojson"'
    return rep


@lru_cache(maxsize=4)
def _heatmap_geojson(dossier, _mtime, brut, quoi):
    cel = HF.centres(charger(dossier, brut=brut)["hex"][1])
    if quoi == "heatmap_flux_points":
        gj = HF.points(cel)
    else:
        E, x0, y0 = HF.surface(cel, HF.RES_M, HF.SIGMA_M, HF.RAYON_M)
        gj = HF.isobandes(E, x0, y0, HF.RES_M)
    return json.dumps(gj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def url_heatmap(data):
    return f"/heatmap/{quote(data['nom'])}/{'brut' if data['brut'] else 'lisse'}.html?v={data['version']}"


def url_couche(data, couche):
    return (f"/geo/{quote(data['nom'])}/{'brut' if data['brut'] else 'lisse'}/{couche}.geojson"
            f"?v={data['version']}")


# ------------------------------------------------------------------ LECTURE DES SORTIES
def lister_rapports():
    """Dossiers de rapports, du plus récent au plus ancien."""
    if not RAPPORTS_DIR.exists():
        return []
    dossiers = [d for d in RAPPORTS_DIR.iterdir()          # archives/ et _en_cours_* n'ont pas de mesures
                if d.is_dir() and (d / "mesures_pas2m.csv").exists()]
    return sorted(dossiers, key=lambda d: d.stat().st_mtime, reverse=True)


def en_bool(v):
    return str(v).strip().lower() in ("true", "1", "oui", "conforme", "ok")


def lire_csv(chemin):
    """CSV GETYS : séparateur ';' et virgule décimale (repli sur le format anglais)."""
    df = pd.read_csv(chemin, sep=";", decimal=",")
    if df.shape[1] == 1:
        df = pd.read_csv(chemin)
    return df


def charger(dossier, brut=False):
    """Données d'un rapport. brut=True : mêmes calculs sur le signal avant lissage du bruit."""
    d = Path(dossier)
    base = _charger(str(d), d.stat().st_mtime)     # le mtime invalide le cache si le rapport change
    if not brut or base["mesures"] is None or "lux_brut" not in base["mesures"]:
        return base
    return _charger_brut(str(d), d.stat().st_mtime)


@lru_cache(maxsize=4)
def _charger(dossier, _mtime):
    d = Path(dossier)
    sortie = {"resume": {}, "hex": None, "mesures": None, "luminaires": None, "aberrants": None,
              "troncons": None, "geoms": None, "erreur_troncons": None, "arbre_pts": None, "brut": False,
              "nom": d.name, "version": int(_mtime)}
    if (d / "resume.json").exists():
        sortie["resume"] = json.loads((d / "resume.json").read_text(encoding="utf-8"))
    if (d / "mesures_pas2m.csv").exists():
        m = lire_csv(d / "mesures_pas2m.csv").rename(columns={"E": "lux", "E_brut": "lux_brut"})
        if "x" not in m:
            m["x"], m["y"] = TO93.transform(m["lon"].to_numpy(), m["lat"].to_numpy())
        m["lux_lisse"] = m["lux"].rolling(5, center=True, min_periods=1).mean()   # évite un point isolé
        sortie["mesures"] = m
        sortie["arbre_pts"] = cKDTree(m[["x", "y"]].to_numpy())

    # --- moyenne au tronçon (calculée une fois puis gardée dans le dossier du rapport)
    if sortie["mesures"] is not None:
        try:
            if (d / "troncons.geojson").exists():
                tr, geoms = T.lire(d)
            else:
                m = sortie["mesures"]
                tr, geoms, _ = T.calculer(m, T.charger_voirie(m["lat"], m["lon"]))
                T.sauver(d, tr, geoms)
            sortie["troncons"], sortie["geoms"] = tr, geoms
        except Exception as e:                           # pas de réseau / Overpass indisponible
            sortie["erreur_troncons"] = str(e)

    # --- hexagones de 2 m : une moyenne par sens de passage, puis moyenne des deux sens
    if sortie["mesures"] is not None:
        sortie["hex"] = H.calculer(sortie["mesures"])
    lum_f = d / "lampadaires_estimes.geojson"                 # analyse actuelle
    if not lum_f.exists():
        lum_f = d / "luminaires_detectes.geojson"            # anciens rapports
    if lum_f.exists():
        gj = json.loads(lum_f.read_text(encoding="utf-8"))
        lum = pd.DataFrame([{**f["properties"], "lon": f["geometry"]["coordinates"][0],
                             "lat": f["geometry"]["coordinates"][1]}
                            for f in gj["features"] if f.get("geometry")])
        sortie["luminaires"] = lum if not lum.empty else None
    sortie["aberrants"] = aberrants_gps(sortie["resume"], sortie["mesures"])
    return sortie


@lru_cache(maxsize=2)
def _charger_brut(dossier, mtime):
    """Variante « avant lissage » : tronçons et hexagones recalculés sur E_brut (même filaire)."""
    base = _charger(dossier, mtime)
    brut = base["mesures"]["lux_brut"]
    # pas de signal brut par capteur : le brut (médiane des 3) est placé aux 3 positions de luxmètre
    m = base["mesures"].assign(lux=brut, L1=brut, L2=brut, L3=brut)
    sortie = {**base, "mesures": m, "brut": True}
    if base["troncons"] is not None:
        tr, geoms, _ = T.calculer(m, T.charger_voirie(m["lat"], m["lon"]))
        cl = base["troncons"].set_index("id")["classe"]
        sortie["troncons"], sortie["geoms"] = tr, geoms
        sortie["troncons"]["classe"] = tr["id"].map(cl).fillna(tr["classe"])
    sortie["hex"] = H.calculer(m)
    return sortie


def _heure_dec(h):
    """'23h31:30' -> 23.525 (heure locale décimale)."""
    try:
        hh, reste = str(h).split("h")
        mm, ss = (reste.split(":") + ["0"])[:2]
        return int(hh) + int(mm) / 60 + float(ss) / 3600
    except ValueError:
        return None


def aberrants_gps(resume, m):
    """Valeurs aberrantes retirées par le filtrage (resume.json -> qualite.aberrants), avec leur point GPS.
    Si l'analyse fournit déjà lat / lon (8 champs), on les prend. Sinon on retrouve le point dans
    mesures_pas2m.csv : parmi les mesures de la même seconde, celle dont le maximum brut (avant filtrage,
    colonne Emax_brut) est le plus fort, c'est-à-dire là où le pic a été vu. Si le véhicule était arrêté
    (pas de mesure à cette seconde), on prend la mesure la plus proche dans le temps (position approchée)."""
    ab = (resume.get("qualite") or {}).get("aberrants") or []
    if not ab:
        return None
    lignes = []
    h_m = m["heure_locale"].to_numpy(float) if m is not None and "heure_locale" in m else None
    emax = m["Emax_brut"].to_numpy(float) if m is not None and "Emax_brut" in m else None
    for k, a in enumerate(ab):
        heure, pic, duree, largeur, fond, motif = (list(a) + [None] * 6)[:6]
        lat = lon = None
        precision = "exacte (GPS de l'analyse)"
        if len(a) >= 8 and a[6] is not None:
            lat, lon = float(a[6]), float(a[7])
        elif h_m is not None and _heure_dec(heure) is not None:
            h = _heure_dec(heure)
            dt = np.abs(((h_m - h + 12) % 24) - 12) * 3600            # écart en secondes (passage de minuit géré)
            fen = np.where(dt <= 1.5)[0]
            if len(fen):
                i = fen[np.argmax(emax[fen])] if emax is not None else fen[np.argmin(dt[fen])]
                precision = "à ± 5 m"
            else:
                i = int(np.argmin(dt))
                precision = "approchée (véhicule à l'arrêt)"
            lat, lon = float(m["lat"].iat[i]), float(m["lon"].iat[i])
        # 9e champ : capteurs qui portent le pic (absent des rapports produits avant son ajout)
        capteur = a[8] if len(a) >= 9 and a[8] else "–"
        lignes.append(dict(id=k, heure=heure, motif=motif, capteur=capteur, pic_lx=pic, fond_lx=fond,
                           largeur_m=largeur, duree_s=duree, lat=lat, lon=lon, precision=precision))
    df = pd.DataFrame(lignes)
    return df if df["lat"].notna().any() else None


def lien_streetview(lat, lon, cap=0):
    """Lien Google Maps officiel qui ouvre Street View au point donné (pas de clé API)."""
    return (f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}"
            f"&heading={cap:.0f}")


def embed_streetview(lat, lon, cap=0):
    """Street View intégrable en iframe, sans clé d'API, orienté dans le sens de la tournée."""
    return f"https://maps.google.com/maps?layer=c&cbll={lat:.6f},{lon:.6f}&cbp=11,{cap:.0f},0,0,0&output=svembed"


def fmt(v, suffixe="", dec=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, (int, float, np.integer, np.floating)):
        v = f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")
    return f"{v}{suffixe}"


def avec_seuils(tr, sdal, rues=None):
    """Ajoute le seuil SDAL et la conformité à chaque tronçon. rues : {nom normalisé: seuil} (fichier SDAL par
    rue), prioritaire sur le seuil du type de voie."""
    tr = tr.copy()
    tr["seuil"] = tr["classe"].map(lambda c: sdal.get(c, T.CLASSES[c][2]))
    if rues:
        propre = tr["nom"].map(lambda n: rues.get(SF.normaliser(n)))
        tr["seuil"] = propre.fillna(tr["seuil"]).astype(float)
    tr["conforme"] = tr["E_moy"] >= tr["seuil"]
    tr["ratio"] = tr["E_moy"] / tr["seuil"]
    tr["a_completer"] = (tr["n_sens"] < 2) & ~tr["sens_unique"].astype(bool)
    return tr


def _mesures_proches(data, tid):
    """Indices des mesures à moins de 15 m de l'axe d'un tronçon."""
    x, y = TO93.transform(*data["geoms"][tid])
    x, y = np.asarray(x), np.asarray(y)
    dd = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    s = np.arange(0, dd[-1] + 1e-9, 4.0)
    return set().union(*data["arbre_pts"].query_ball_point(
        np.c_[np.interp(s, dd, x), np.interp(s, dd, y)], r=T.RAYON_M))


def point_le_plus_sombre(data, *tids):
    """Point de mesure le plus sombre (lissé) le long d'un ou plusieurs tronçons -> (lat, lon, cap, lux)."""
    proches = sorted(set().union(*(_mesures_proches(data, t) for t in tids)))
    if not proches:
        glon, glat = data["geoms"][tids[0]]
        return float(np.mean(glat)), float(np.mean(glon)), 0.0, None
    p = data["mesures"].iloc[proches].sort_values("lux_lisse").iloc[0]
    return float(p["lat"]), float(p["lon"]), float(p.get("cap", 0) or 0), float(p["lux_lisse"])


def cap_proche(data, lat, lon):
    """Cap du véhicule à la mesure la plus proche (oriente Street View dans le sens de la rue)."""
    if data["mesures"] is None:
        return 0.0
    x, y = TO93.transform(lon, lat)
    _, i = data["arbre_pts"].query([x, y])
    return float(data["mesures"].iloc[i].get("cap", 0) or 0)


# ------------------------------------------------------------------ COMPOSANTS
def kpi(icone, titre, valeur, detail=None, couleur="light", barre=None):
    corps = [
        html.Div([html.I(className=f"bi bi-{icone} kpi-icone text-{couleur}"),
                  html.Span(titre, className="kpi-titre")], className="d-flex align-items-center gap-2"),
        html.Div(valeur, className=f"kpi-valeur text-{couleur}"),
    ]
    if barre is not None:
        corps.append(dbc.Progress(value=barre, color=couleur, className="kpi-barre"))
    if detail:
        corps.append(html.Div(detail, className="kpi-detail"))
    return dbc.Col(html.Div(corps, className="kpi"), xs=6, xl=3)


def carte_vide(msg, h=CARTE_H):
    fig = go.Figure()
    fig.update_layout(annotations=[dict(text=msg, showarrow=False, font_size=15, font_color="#6B7280")],
                      xaxis_visible=False, yaxis_visible=False, height=h,
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ------------------------------------------------------------------ CARTE (Leaflet)
NS = "getys"                                      # fonctions JS de assets/carte_leaflet.js


def fn(nom):
    return {"variable": f"{NS}.{nom}"}


def legende(mode, couleur, couches, data):
    """Légende posée sur la carte (Leaflet n'en fournit pas)."""
    blocs = []
    if mode == "heatmap":                          # la page de la heatmap a sa propre légende
        return blocs
    if mode == "troncons" and couleur == "conformite":
        blocs.append(html.Div([html.Div([html.Span(className="pastille", style={"background": c}), t])
                               for t, c in ((f"Plus de {RA.LUX_ROUGE} lx", RA.ROUGE_FORT),
                                            (f"Conforme, jusqu'à {RA.LUX_ROUGE} lx", RA.CONFORME),
                                            ("Sous le seuil", RA.SOUS), ("Sous le seuil · 1 seul sens", RA.SOUS_1S))]))
    elif mode == "troncons":
        blocs.append(html.Div([html.Div([html.Span(className="pastille", style={"background": c}),
                                         f"{PALIERS[i]:g}–{PALIERS[i + 1]:g} lx" if PALIERS[i + 1] < math.inf
                                         else f"> {PALIERS[i]:g} lx"])
                               for i, c in enumerate(COUL_PALIERS)]))
    else:
        rampe = ", ".join(f"rgb{c} {2 * v}%" for v, c in STOPS)
        blocs.append(html.Div([html.Div("lux", className="leg-titre"),
                               html.Div(className="rampe", style={"background": f"linear-gradient(90deg, {rampe})"}),
                               html.Div([html.Span("0"), html.Span("25"), html.Span("50")], className="rampe-graduations")]))
    ab = data["aberrants"] if data else None
    if "ab" in (couches or []) and ab is not None:
        blocs.append(html.Div([html.Div("Valeurs aberrantes retirées", className="leg-titre mt-1")] +
                              [html.Div([html.Span(className="pastille rond", style={"background": COUL_MOTIFS.get(t, ROUGE)}),
                                         f"{t} ({n})"]) for t, n in ab["motif"].value_counts().items()]))
    return blocs


def marqueurs_aberrants(ab):
    """Couche « Valeurs aberrantes » : un cercle par pic retiré (25 au plus), cliquable -> Street View."""
    if ab is None:
        return []
    return [dl.CircleMarker(id={"type": "aberrant", "k": int(a.id)}, center=[a.lat, a.lon], radius=7, weight=2,
                            color="#111", fillColor=COUL_MOTIFS.get(a.motif, ROUGE), fillOpacity=0.95,
                            children=dl.Tooltip(f"Valeur aberrante retirée — {a.motif} · pic {fmt(a.pic_lx, ' lx', 0)} "
                                                f"(fond {fmt(a.fond_lx, ' lx', 0)}) · {a.heure}"))
            for a in ab.dropna(subset=["lat"]).itertuples()]


VIDE = {"type": "FeatureCollection", "features": []}


# ------------------------------------------------------------------ PANNEAUX
def vue_types(tr):
    """Par type de voie : une carte par type (seuil, linéaire, part conforme, éclairement moyen) et le linéaire
    conforme / non conforme / mesuré dans un seul sens, en km."""
    if tr is None or tr.empty:
        return html.P("Pas de tronçons dans ce rapport.", className="text-muted")
    lignes, cartes = [], []
    for c in [c for c in T.CLASSES if c in set(tr["classe"])]:
        t = tr[tr["classe"] == c]
        lg = t["longueur_m"]
        km = lg.sum() / 1000
        conf = lg[t["conforme"]].sum() / 1000
        nc_1s = lg[~t["conforme"] & t["a_completer"]].sum() / 1000
        nc = km - conf - nc_1s
        e = float((t["E_moy"] * lg).sum() / lg.sum())
        seuil = float(t["seuil"].iat[0])
        part = 100 * conf / km if km else 0
        coul = "success" if part >= 80 else "warning" if part >= 50 else "danger"
        lignes.append(dict(type=T.CLASSES[c][0], conf=conf, nc=nc, nc_1s=nc_1s))
        cartes.append(dbc.Col(xs=12, md=6, children=html.Div(className="carte-type", children=[
            html.Div([html.Span(T.CLASSES[c][0], className="carte-type-nom"),
                      html.Span(f"seuil {fmt(seuil, ' lx')}", className="badge-type")],
                     className="d-flex justify-content-between align-items-center"),
            html.Div([html.Span(fmt(part, " %", 0), className=f"carte-type-part text-{coul}"),
                      html.Span(" du linéaire conforme", className="carte-type-legende")]),
            dbc.Progress(value=part, color=coul, className="kpi-barre"),
            html.Div(className="carte-type-chiffres", children=[
                html.Div([html.B(fmt(e, " lx")), html.Span("éclairement moyen")],
                         className="text-danger" if e < seuil else ""),
                html.Div([html.B(fmt(km, " km", 2)), html.Span(f"{len(t)} tronçon(s)")]),
                html.Div([html.B(f"{int((~t['conforme']).sum())}"), html.Span("sous le seuil")]),
            ]),
        ])))
    d = pd.DataFrame(lignes).iloc[::-1]
    fig = go.Figure()
    for col, nom, coul in (("conf", "Conforme", VERT), ("nc", "Non conforme", ROUGE),
                           ("nc_1s", "Non conforme · 1 seul sens", ORANGE)):
        fig.add_trace(go.Bar(y=d["type"], x=d[col], name=nom, orientation="h", marker_color=coul,
                             marker_line_width=0, hovertemplate="%{y}<br>" + nom + " : %{x:.2f} km<extra></extra>"))
    fig.update_layout(barmode="stack", height=60 + 52 * len(d), margin=dict(l=10, r=10, t=30, b=30),
                      xaxis_title="Linéaire mesuré (km)", bargap=0.35,
                      legend=dict(orientation="h", y=1.15, x=0, font_size=11, traceorder="normal"))
    return dbc.Row(className="g-3", children=[
        dbc.Col(xl=6, children=dbc.Row(cartes, className="g-2")),
        dbc.Col(xl=6, children=dcc.Graph(figure=fig, config={"displayModeBar": False})),
    ])


def types_lampes(data):
    """Parc estimé par type de lampe : nombre et puissance installée."""
    ty = data["resume"].get("types")
    if not ty:
        return carte_vide("Pas d'estimation des lampadaires dans ce rapport", 360)
    t = pd.DataFrame(ty).sort_values("n")
    lib = [x.split(":")[0].split("(")[0].strip() for x in t["libelle"]]
    fig = go.Figure(go.Bar(
        x=t["n"], y=lib, orientation="h", marker_color=[COUL_TYPES.get(x, "#fff") for x in t["type"]],
        customdata=np.c_[t["libelle"], t["p_med"], t["p_tot_kw"], t["c_tot_kw"]],
        text=[f"{n} · {p:.0f} kW" for n, p in zip(t["n"], t["p_tot_kw"])], textposition="outside",
        hovertemplate="<b>%{customdata[0]}</b><br>%{x} lampadaires · %{customdata[1]} W médian"
                      "<br>Puissance installée : %{customdata[2]} kW · consommation %{customdata[3]} kW"
                      "<extra></extra>"))
    k = data["resume"].get("kpi", {})
    fig.update_layout(height=360, margin=dict(l=10, r=40, t=30, b=30), xaxis_title="nombre de lampadaires estimés",
                      title=dict(text=f"{fmt(k.get('n_lampadaires'), dec=0)} lampadaires estimés · "
                                      f"{fmt(k.get('puissance_tot_kw'), ' kW')} installés", font_size=13, x=0.01))
    return fig


def bruit(data):
    """Onglet « Bruit et valeurs aberrantes » : ce que le filtrage a corrigé et la liste des pics retirés,
    chacun avec ses coordonnées GPS. Clic sur une ligne -> Street View à droite et carte centrée sur le point."""
    q, k = data["resume"].get("qualite", {}), data["resume"].get("kpi", {})
    ab = data["aberrants"]
    cols = [("heure", "Heure"), ("motif", "Motif"), ("capteur", "Capteur"), ("pic_lx", "Pic (lx)"), ("fond_lx", "Fond (lx)"),
            ("largeur_m", "Largeur (m)"), ("gps", "Coordonnées GPS"), ("precision", "Position")]
    # table toujours présente (vide et masquée s'il n'y a rien) : elle est une entrée de choisir_point
    table_vide = html.Div(dash_table.DataTable(id="table-aberrants", data=[],
                                               columns=[{"name": n, "id": c} for c, n in cols]), hidden=True)
    if not q:
        return html.Div([html.P("Pas d'informations sur le bruit dans ce rapport.", className="text-muted"),
                         table_vide])
    par_motif = ab["motif"].value_counts().to_dict() if ab is not None else {}
    pastille = lambda c: html.Span(className="pastille rond", style={"background": c})  # noqa: E731
    intro = html.Div([
        html.P([html.B(f"{fmt(k.get('n_aberrants'), dec=0)} valeurs aberrantes retirées"),
                f" ({fmt(q.get('part_aberrants'), ' %', 2)} des mesures)"],
               className="small mb-2"),
        html.Div([html.Span([pastille(COUL_MOTIFS.get(m_, ROUGE)), f"{n} {m_}"], className="me-3 small")
                  for m_, n in par_motif.items()], className="mb-2"),
    ])
    if ab is None:
        return html.Div([intro, table_vide])
    t = ab.assign(gps=[f"{a:.5f}, {b:.5f}" if pd.notna(a) else "–" for a, b in zip(ab["lat"], ab["lon"])])
    t = t.astype(object).where(t.notna(), None)
    return html.Div([intro, dash_table.DataTable(
        id="table-aberrants", data=t[[c for c, _ in cols] + ["id"]].to_dict("records"),
        columns=[{"name": n, "id": c} for c, n in cols], sort_action="native", page_size=10, **STYLE_TABLE,
        style_data_conditional=[
            *[{"if": {"filter_query": f'{{motif}} = "{m_}"', "column_id": "motif"}, "color": c}
              for m_, c in COUL_MOTIFS.items()],
            {"if": {"column_id": "gps"}, "fontFamily": "monospace"},
            {"if": {"state": "active"}, "backgroundColor": "rgba(62,156,97,0.12)", "border": "1px solid #3E9C61"}],
    )])


def chronologie(data):
    ts = data["resume"].get("ts")
    if not ts:
        return carte_vide("Pas de série temporelle dans ce rapport", 360)
    t = pd.DataFrame(ts, columns=["h", "lux"])
    hh = t["h"].apply(lambda h: f"{int(h)}h{int(round((h % 1) * 60)) % 60:02d}")
    fig = go.Figure(go.Scatter(x=t["h"], y=t["lux"], mode="lines", line=dict(color=AMBRE, width=2),
                               fill="tozeroy", fillcolor="rgba(0,147,187,0.12)", text=hh,
                               hovertemplate="%{text} — %{y:.1f} lx<extra></extra>"))
    pas = list(range(int(t["h"].min()), int(t["h"].max()) + 2))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=30), yaxis_title="lux moyen (1 min)",
                      hovermode="x",                     # clic n'importe où à la verticale d'une heure
                      xaxis=dict(tickvals=pas, ticktext=[f"{h}h" for h in pas], title="heure locale"))
    return fig


STYLE_TABLE = dict(
    style_table={"overflowX": "auto", "minWidth": "100%"},
    style_header={"backgroundColor": VERT_FONCE, "color": "#FFFFFF", "fontWeight": "700",
                  "border": "none", "borderBottom": "2px solid " + LIME},
    style_filter={"backgroundColor": "var(--surface-2)", "color": "var(--texte)"},
    style_cell={"backgroundColor": "transparent", "color": "var(--texte)", "border": "none",
                "borderBottom": "1px solid var(--bord)", "textAlign": "left", "padding": "7px 10px",
                "fontFamily": "Roboto, Arial, sans-serif", "fontSize": "13px", "cursor": "pointer",
                "whiteSpace": "normal", "height": "auto"},
    style_cell_conditional=[{"if": {"column_id": "nom"}, "minWidth": "190px", "fontWeight": "500"},
                            {"if": {"column_id": "detail"}, "minWidth": "220px"}],
)


def tableau_troncons(tr):
    if tr is None or tr.empty:
        return html.P("Pas de tronçons dans ce rapport.", className="text-muted")
    r = tr.assign(
        type=tr["classe"].map(lambda c: T.CLASSES[c][0]),
        sens=np.where(tr["sens_unique"], "sens unique", tr["n_sens"].astype(str) + " / 2"),
        etat=np.where(tr["conforme"], "✅", np.where(tr["a_completer"], "⚠️ ❌", "❌")),
    ).sort_values("ratio")
    r = r.astype(object).where(r.notna(), None)          # NaN -> null (sens non mesuré)
    cols = [("nom", "Voie"), ("type", "Type"), ("longueur_m", "Long. (m)"), ("sens", "Sens mesurés"),
            ("n_passages", "Passages"), ("E_sens1", "E sens 1"), ("E_sens2", "E sens 2"),
            ("E_moy", "E moy. (lx)"), ("U0", "U0"), ("seuil", "Seuil"), ("etat", "Conforme")]
    return html.Div([
        dash_table.DataTable(
            id="table-troncons", data=r.to_dict("records"), columns=[{"name": n, "id": c} for c, n in cols],
            sort_action="native", filter_action="native", page_size=12, **STYLE_TABLE,
            style_data_conditional=[
                {"if": {"filter_query": '{conforme} = false'}, "backgroundColor": "rgba(231,81,19,0.08)"},
                {"if": {"state": "active"}, "backgroundColor": "rgba(62,156,97,0.12)", "border": "1px solid #3E9C61"}],
        ),
    ])


def troncons_a_fiabiliser(tr):
    """Tronçons dont la moyenne est fragile : un seul sens, couverture faible, passages incohérents."""
    if tr is None or tr.empty:
        return []
    lignes = []
    for r in tr.itertuples():
        if r.a_completer:
            lignes.append((r, "Un seul sens mesuré", f"sens 1 : {fmt(r.E_sens1, ' lx')} · sens 2 : {fmt(r.E_sens2, ' lx')}"))
        if r.couverture < 60:
            lignes.append((r, "Couverture faible", f"{r.couverture} % de la longueur mesurée"))
        if r.n_passages >= 3 and r.cv_passages > 35:
            lignes.append((r, "Passages incohérents", f"écart entre passages {r.cv_passages} % ({r.n_passages} passages)"))
    return [{"id": f"{r.id}|{k}", "nom": r.nom, "probleme": p, "detail": d, "E_moy": r.E_moy,
             "longueur_m": r.longueur_m} for k, (r, p, d) in enumerate(lignes)]


def qualite(data, tr):
    q, k = data["resume"].get("qualite", {}), data["resume"].get("kpi", {})
    lignes = [
        ("Source GNSS", q.get("gnss_source")),
        ("Positions GNSS", fmt(q.get("n_pos"), dec=0)),
        ("HDOP médian / satellites", f"{fmt(q.get('hdop_med'))} / {fmt(q.get('nsat_med'), dec=0)}"),
        ("Décalage d'horloge corrigé", fmt(q.get("decalage_horloge_s"), " s")),
        ("Mesures négatives", fmt(q.get("part_negatifs"), " %", 2)),
        ("Points aberrants écartés", f"{fmt(k.get('n_aberrants'), dec=0)} ({fmt(q.get('part_aberrants'), ' %', 2)})"),
        ("Mailles vues plusieurs fois", f"{fmt(k.get('n_multi', k.get('n_hex_multi')), dec=0)} — CV médian "
                                        f"{fmt(k.get('cv_maille_med', k.get('cv_multi_med')), ' %', 0)}"),
    ]
    m = data["mesures"]
    if m is not None and "lux_brut" in m:
        d = (m["lux_brut"] - m["lux"]).abs()
        lignes.append(("Effet du lissage du bruit", f"{fmt(100 * (d > 0.5).mean(), ' %', 1)} des points modifiés · "
                                                   f"écart max {fmt(d.max(), ' lx', 0)}"))
    cal = data["resume"].get("calage") or {}
    if cal.get("ok"):
        lignes.append(("Calage capteurs / GPS (auto)",
                       f"décalage {fmt(cal.get('decalage_m'), ' m', 2)} · latence {fmt(cal.get('latence_s'), ' s', 2)}"))
    if tr is not None:
        lignes += [("Tronçons de voirie (OSM) mesurés", fmt(len(tr), dec=0)),
                   ("Écart médian entre les deux sens", fmt((tr["E_sens1"] - tr["E_sens2"]).abs().median(), " lx"))]
    fiab = troncons_a_fiabiliser(tr)
    return html.Div([
        dbc.Table(html.Tbody([html.Tr([html.Td(a, className="text-muted"), html.Td(b or "–")]) for a, b in lignes]),
                  size="sm", className="mb-3 tableau-qualite"),
        html.H6([html.I(className="bi bi-exclamation-diamond me-2 text-warning"),
                 f"Tronçons à fiabiliser ({len(fiab)})"], className="fw-semibold"),
        dash_table.DataTable(
            id="table-qualite", data=fiab, page_size=8, sort_action="native", filter_action="native",
            columns=[{"name": "Voie", "id": "nom"}, {"name": "Problème", "id": "probleme"},
                     {"name": "Détail", "id": "detail"}, {"name": "E moy. (lx)", "id": "E_moy"},
                     {"name": "Long. (m)", "id": "longueur_m"}], **STYLE_TABLE,
            style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "rgba(0,147,187,0.12)",
                                     "border": "1px solid #3E9C61"}]),
    ])


# ------------------------------------------------------------------ MISE EN PAGE
def carte_bloc(titre, icone, corps, droite=None, **kw):
    return dbc.Card([
        dbc.CardHeader(html.Div([html.Span([html.I(className=f"bi bi-{icone} me-2"), titre], className="bloc-titre"),
                                 droite], className="d-flex justify-content-between align-items-center gap-2")),
        dbc.CardBody(corps, **kw),
    ], className="bloc")


def segmente(id_, valeur, options, titre=None):
    radio = dbc.RadioItems(id=id_, value=valeur, className="btn-group btn-group-sm segmente",
                           inputClassName="btn-check", labelClassName="btn btn-outline-secondary",
                           labelCheckedClassName="active", options=options)
    if not titre:
        return radio
    return html.Div([html.Span(titre, className="outil-titre"), radio], className="outil", id=f"outil-{id_}")


# fond de carte : même choix (et même valeur, synchronisée) dans la vue Analyste et la vue Opérateur
OPTIONS_FOND = [
    {"label": [html.I(className="bi bi-moon-stars me-1"), "Sombre"], "value": "sombre"},
    {"label": [html.I(className="bi bi-sun me-1"), "Clair"], "value": "clair"},
    {"label": [html.I(className="bi bi-globe-europe-africa me-1"), "Satellite"], "value": "satellite"}]

upload_style = {"border": "1.5px dashed #3d444d", "borderRadius": "10px", "padding": "22px",
                "textAlign": "center", "cursor": "pointer"}

champs_sdal = [
    dbc.Row(className="g-2 align-items-center mb-1", children=[
        dbc.Col(html.Span(lib, className="small"), width=7),
        dbc.Col(dbc.InputGroup([dbc.Input(id=f"sdal-{c}", type="number", value=v, min=0, step=0.5, debounce=True,
                                          size="sm", className="text-end"),
                                dbc.InputGroupText("lx", className="py-0 small")], size="sm"), width=5),
    ]) for c, (lib, _, v) in T.CLASSES.items()]

def legende_operateur():
    rampe = ", ".join(f"rgb{c} {2 * v}%" for v, c in STOPS)
    return [html.Div("3 capteurs · lux bruts", className="leg-titre"),
            html.Div(className="rampe", style={"background": f"linear-gradient(90deg, {rampe})"}),
            html.Div([html.Span("0"), html.Span("25"), html.Span("50+")], className="rampe-graduations"),
            html.Div("3 rangées dans le sens de marche : L3 à gauche, L2 au centre, L1 à droite"),
            html.Div("écartement agrandi quand on dézoome", className="text-muted")]


def page_operateur():
    """Vue Opérateur : le .tdms brut sur la carte (sans analyse), contrôles, points annotés et notes."""
    return dbc.Container(fluid=True, className="contenu", children=[
        dbc.Row(className="g-3", children=[
            dbc.Col(xl=8, children=dbc.Card(className="bloc", children=[
                dbc.CardHeader(html.Div(className="barre-outils", children=[
                    html.Div(className="outil", children=[
                        html.Span("Fichier .tdms de la tournée", className="outil-titre"),
                        html.Span(id="op-nom", className="fw-semibold")]),
                    segmente("op-fond", "clair", OPTIONS_FOND, "Fond"),
                    dbc.Checklist(id="op-mode-note", value=[], switch=True, className="small outil-couches",
                                  options=[{"label": "Ajouter un point au clic", "value": "on"}]),
                ])),
                dcc.Store(id="op-fichier"), html.Div(id="op-etat"),
                dcc.Store(id="op-trace-url"), html.Div(id="op-trace-dessin", hidden=True),
                dbc.CardBody(html.Div(id="op-cadre", className="carte-leaflet fond-clair", children=[
                    # zoom par paliers entiers : les images satellite sont affichées à leur taille exacte
                    dl.Map(id="op-carte", center=[48.785, 2.54], zoom=14, zoomSnap=1, preferCanvas=True,
                           maxZoom=ZOOM_MAX, style={"height": f"{CARTE_H}px", "width": "100%"}, children=[
                               dl.LayerGroup(id="op-fonds", children=tuiles("clair", "op-tuiles")),
                               dl.LayerGroup(id="op-marqueurs"),
                               dl.ScaleControl(position="bottomright", imperial=False)]),
                    html.Div(legende_operateur(), className="legende-carte")]), className="p-0"),
            ])),
            dbc.Col(xl=4, className="d-flex flex-column gap-3", children=[
                carte_bloc("Contrôles de la tournée", "clipboard-check", html.Div(id="op-controles")),
                carte_bloc("Points annotés", "pin-map", [
                    html.Div(id="op-form", style={"display": "none"}, className="mb-2", children=[
                        html.Div(id="op-coord", className="small text-muted mb-1"), dcc.Store(id="op-attente"),
                        dbc.Textarea(id="op-texte", rows=2, placeholder="Commentaire (ex. lampadaire éteint, "
                                     "travaux, rue non parcourue…)", className="mb-2"),
                        dbc.Button([html.I(className="bi bi-plus-lg me-1"), "Ajouter le point"], id="op-ajouter",
                                   size="sm", color="success")]),
                    html.Div(id="op-liste"), html.Div(id="op-aller-fait", hidden=True),
                ]),
                carte_bloc("Notes de la tournée", "journal-text", [
                    dbc.Textarea(id="op-notes", rows=5, placeholder="Conditions, incidents, secteurs à repasser…"),
                    html.Div(className="d-flex align-items-center gap-2 mt-2", children=[
                        dbc.Button("Enregistrer", id="op-enregistrer", size="sm", color="success"),
                        html.Span(id="op-notes-etat", className="small text-muted")])]),
            ]),
        ]),
    ])


def page_affaires():
    """Vue Chargé d'affaires : rapport à composer (sections cochées) pour la tournée choisie en haut."""
    return dbc.Container(fluid=True, className="contenu", children=[
        html.Div(className="contexte", children=[
            html.I(className="bi bi-info-circle me-1"),
            "Rapport de la tournée choisie en haut de la page. Une première sélection est proposée : cochez ce que vous "
            "voulez y mettre, ajoutez vos informations, puis téléchargez-le (PDF : Fichier → Imprimer depuis le "
            "navigateur)."]),
        dbc.Row(className="g-3", children=[
            dbc.Col(xl=4, className="d-flex flex-column gap-3", children=[
                carte_bloc("Contenu du rapport", "list-check", [
                    dbc.Checklist(id="ra-sections", value=RA.DEFAUT, className="small",
                                  options=[{"label": lib, "value": cle} for cle, lib, _ in RA.SECTIONS]),
                    dbc.Button("Revenir à la sélection proposée", id="ra-defaut", size="sm", color="link",
                               className="px-0 mt-1")]),
                carte_bloc("Informations", "pencil-square", [
                    dbc.Label("Titre", className="small mb-0"),
                    dbc.Input(id="ra-titre", size="sm", placeholder="Diagnostic de l'éclairage public — <commune>"),
                    dbc.Label("Destinataire", className="small mb-0 mt-2"),
                    dbc.Input(id="ra-destinataire", size="sm", placeholder="Mairie de …"),
                    dbc.Label("Établi par", className="small mb-0 mt-2"),
                    dbc.Input(id="ra-auteur", size="sm", placeholder="Nom, Bouygues Energies & Services"),
                    dbc.Label("Commentaire / recommandations", className="small mb-0 mt-2"),
                    dbc.Textarea(id="ra-commentaire", rows=5),
                    html.Div(className="d-flex gap-2 mt-3", children=[
                        dbc.Button([html.I(className="bi bi-eye me-1"), "Aperçu"], id="ra-apercu", size="sm",
                                   color="success", outline=True),
                        dbc.Button([html.I(className="bi bi-download me-1"), "Télécharger le rapport"],
                                   id="ra-telecharger", size="sm", color="success")]),
                    dcc.Download(id="ra-download")]),
            ]),
            dbc.Col(xl=8, children=carte_bloc("Aperçu du rapport", "file-earmark-richtext", dcc.Loading(
                html.Iframe(id="ra-cadre", className="cadre-rapport"), type="dot", color=AMBRE), className="p-0")),
        ]),
    ])


def page_doc():
    """Onglet Doc : techniques et algorithmes, stack technique, explication détaillée du code."""
    complet = DOC.texte_complet()
    return dbc.Container(fluid=True, className="contenu doc", children=dbc.Row(className="g-3", children=[
        dbc.Col(xl=7, children=carte_bloc("Techniques et algorithmes", "cpu", dcc.Markdown(DOC.ALGORITHMES))),
        dbc.Col(xl=5, className="d-flex flex-column gap-3", children=[
            carte_bloc("Stack technique", "stack", dcc.Markdown(DOC.STACK)),
            carte_bloc("Explication détaillée du code", "journal-code", html.Details([
                html.Summary("EXPLICATION_DU_CODE.md"), dcc.Markdown(complet)]) if complet else
                html.P("Fichier EXPLICATION_DU_CODE.md absent.", className="text-muted small")),
        ]),
    ]))


app.layout = html.Div(className="page", children=[
    dcc.Store(id="selection"), html.Div(id="defilement", hidden=True),
    dcc.Store(id="theme", storage_type="local", data="clair"), html.Div(id="theme-applique", hidden=True),
    dcc.Store(id="urls-prechargement"), html.Div(id="prechargement", hidden=True),
    html.Div(id="fond-vectoriel", hidden=True),
    # voiles de chargement : ouverture d'une tournée / analyse d'une nouvelle tournée
    html.Div(id="voile-chargement", className="voile", style={"display": "none"}, children=html.Div(
        className="voile-boite", children=[html.Div(className="voile-anneau"),
                                           html.Div("Chargement de la tournée…", className="voile-titre"),
                                           html.Div("indicateurs, tronçons et carte", className="voile-detail")])),
    html.Div(id="voile-analyse", className="voile", style={"display": "none"}, children=html.Div(
        className="voile-boite", children=[html.Div(className="voile-anneau"),
                                           html.Div("Analyse de la tournée en cours…", className="voile-titre"),
                                           html.Div("lecture du .tdms, filtrage, lampadaires : environ 10 s",
                                                    className="voile-detail")])),
    # --- barre du haut
    html.Header(className="entete", children=dbc.Container(fluid=True, children=dbc.Row(
        align="center", className="g-2", children=[
            dbc.Col(xs=12, lg=True, children=html.Div([
                # logos repris du dossier du hackathon (Getys_Hackathon.pdf)
                html.Img(src=app.get_asset_url("logo-getys.jpg"), alt="GETYS — Relevé & visualisé !",
                         className="logo-getys"),
                html.Div([html.H1("Clair-Obscur", className="titre mb-0"),
                          html.Div("Tableau de bord de l'éclairage public", className="sous-titre")]),
                html.A(html.Img(src=app.get_asset_url("logo-bouygues-es.png"), alt="Bouygues Energies & Services"),
                       href=LIEN_HACKATHON, target="_blank", rel="noopener", className="logo-entreprise",
                       title="Bouygues Energies & Services"),
            ], className="d-flex align-items-center gap-3")),
            dbc.Col(xs=12, md=6, lg=4, children=dcc.Dropdown(
                id="choix-rapport", clearable=False, placeholder="Choisir une tournée…",
                className="choix-rapport")),
            dbc.Col(xs="auto", children=html.A([html.I(className="bi bi-globe-europe-africa me-1"),
                                                 "Carte satellite"], id="lien-carte", target="_blank",
                                                className="btn btn-outline-light btn-sm disabled")),
            dbc.Col(xs="auto", children=dbc.Button([html.I(className="bi bi-cloud-arrow-up me-1"),
                                                     "Nouvelle tournée"], id="btn-nouvelle",
                                                    color="warning", size="sm", className="fw-semibold")),
            dbc.Col(xs="auto", children=html.Button(html.I(id="icone-theme", className="bi bi-moon-stars"),
                                                    id="btn-theme", className="btn btn-outline-light btn-sm bouton-theme",
                                                    title="Mode sombre", **{"aria-label": "Changer de mode clair / sombre"})),
        ]))),

    # --- vues par rôle
    html.Nav(className="barre-pages", children=dbc.Container(fluid=True, children=dbc.RadioItems(
        id="page", value="analyste", persistence=True, persistence_type="session",
        className="btn-group btn-group-sm segmente pages", inputClassName="btn-check",
        labelClassName="btn btn-outline-secondary", labelCheckedClassName="active", options=[
            {"label": [html.I(className="bi bi-truck me-1"), "Opérateur"], "value": "operateur"},
            {"label": [html.I(className="bi bi-graph-up me-1"), "Analyste"], "value": "analyste"},
            {"label": [html.I(className="bi bi-briefcase me-1"), "Chargé d'affaires"], "value": "affaires"},
            {"label": [html.I(className="bi bi-book me-1"), "Doc"], "value": "doc"}]))),
    dcc.Store(id="sdal-rues", data={}),

    html.Div(id="page-operateur", style={"display": "none"}, children=page_operateur()),
    html.Div(id="page-affaires", style={"display": "none"}, children=page_affaires()),
    html.Div(id="page-doc", style={"display": "none"}, children=page_doc()),

    html.Div(id="page-analyste", children=dbc.Container(fluid=True, className="contenu", children=[
        html.Div(id="contexte", className="contexte"),
        dcc.Loading(html.Div(id="message"), type="dot", color=AMBRE),
        dbc.Row(id="kpis", className="g-3 mb-3"),

        dbc.Row(className="g-3", children=[
            # --- carte
            dbc.Col(xl=8, children=dbc.Card(id="bloc-carte", className="bloc", children=[
                dbc.CardHeader(html.Div(className="barre-outils", children=[
                    segmente("mode-carte", "troncons", [
                        {"label": [html.I(className="bi bi-bezier2 me-1"), "Tronçons"], "value": "troncons"},
                        {"label": [html.I(className="bi bi-hexagon me-1"), "Hexagones"], "value": "hex"},
                        {"label": [html.I(className="bi bi-thermometer-half me-1"), "Heatmap"],
                         "value": "heatmap"}], "Vue"),
                    segmente("couleur-carte", "eclairement", [
                        {"label": "Éclairement", "value": "eclairement"},
                        {"label": "Conformité SDAL", "value": "conformite"}], "Couleur (tronçons)"),
                    segmente("lissage", "apres", [
                        {"label": [html.I(className="bi bi-soundwave me-1"), "Avant"], "value": "avant"},
                        {"label": [html.I(className="bi bi-activity me-1"), "Après lissage"], "value": "apres"}],
                        "Bruit (tronçons, hexagones)"),
                    segmente("fond", "clair", OPTIONS_FOND, "Fond"),
                    html.Div(id="outil-export", className="outil", style={"display": "none"}, children=[
                        html.Span("Export", className="outil-titre"),
                        dbc.DropdownMenu(label=[html.I(className="bi bi-download me-1"), "Exporter en GeoJSON"],
                                         size="sm", color="light", className="menu-export", children=[
                            dbc.DropdownMenuItem([html.Div("Zones par classe de lux", className="fw-semibold"),
                                                  html.Div("surface lissée : un MultiPolygon par classe",
                                                           className="small text-muted")],
                                                 id="export-zones", external_link=True),
                            dbc.DropdownMenuItem([html.Div("Cellules de 2 m", className="fw-semibold"),
                                                  html.Div("points : moyenne et valeur de chaque sens",
                                                           className="small text-muted")],
                                                 id="export-cellules", external_link=True),
                        ])]),
                    dbc.Checklist(id="couches", value=[], switch=True, className="small outil-couches",
                                  options=[{"label": "Valeurs aberrantes", "value": "ab"}]),
                ])),
                html.Div(id="bandeau-lissage"),
                dbc.CardBody(html.Div(id="cadre-carte", className="carte-leaflet fond-clair", children=[
                    dl.Map(id="carte", center=[48.785, 2.54], zoom=14.5, zoomSnap=0.25, zoomDelta=0.5,
                           preferCanvas=True, maxZoom=ZOOM_MAX, style={"height": f"{CARTE_H}px", "width": "100%"},
                           children=[
                               dl.LayerGroup(id="fonds", children=tuiles("clair")),
                               dl.GeoJSON(id="geo-troncons", data=VIDE, style=fn("styleTroncon"),
                                          onEachFeature=fn("infoTroncon"), hideout={}),
                               dl.GeoJSON(id="geo-hex", data=VIDE, style=fn("styleHex"), onEachFeature=fn("infoHex")),
                               dl.LayerGroup(id="groupe-aberrants"),
                               dl.LayerGroup(id="groupe-selection"),
                               dl.ScaleControl(position="bottomright", imperial=False),
                           ]),
                    html.Div(id="legende", className="legende-carte"),
                    # vue Heatmap : page de heatmap_flux.py posée sur la carte (fond, vue, lampadaires intégrés)
                    html.Iframe(id="carte-heatmap", className="carte-heatmap", style={"display": "none"},
                                title="Heatmap de l'éclairement"),
                ]), className="p-0"),
            ])),
            # --- Street View + SDAL
            dbc.Col(xl=4, className="d-flex flex-column gap-3", children=[
                carte_bloc("Street View", "camera", [
                    html.Div(id="sv-cadre", className="sv-cadre", children=html.Div(
                        [html.I(className="bi bi-geo-alt fs-2 d-block mb-2"),
                         "Sélectionnez un point ou un tronçon pour voir la rue."], className="sv-vide")),
                    html.Div(id="sv-legende", className="sv-legende"),
                ], className="p-0", droite=html.A(id="sv-externe", target="_blank",
                                                  className="small text-info d-none",
                                                  children=[html.I(className="bi bi-box-arrow-up-right me-1"),
                                                            "Google Maps"])),
                carte_bloc("SDAL — seuils par type de voie", "sliders", [
                    *champs_sdal,
                    html.Hr(className="my-2"),
                    dcc.Upload(id="sdal-upload", accept=".csv,.txt", className="sdal-upload", children=html.Div(
                        [html.I(className="bi bi-filetype-csv me-1"), "Importer un fichier SDAL (.csv)"])),
                    html.Div(className="small text-muted mt-1", children=[
                        "Par type de voie (colonnes ", html.Code("type;seuil"), ") ou par rue (", html.Code("rue;seuil"),
                        ") : le seuil d'une rue remplace celui de son type."]),
                    html.Div(id="sdal-message", className="small mt-1"),
                ]),
            ]),
        ]),

        dbc.Row(className="g-3 mt-0", children=[
            dbc.Col(xs=12, children=dbc.Card(className="bloc h-100", children=[
                dbc.CardHeader(dbc.Tabs(id="onglets", active_tab="troncons", className="card-header-tabs", children=[
                    dbc.Tab(label="Tronçons", tab_id="troncons"),
                    dbc.Tab(label="Par type de voie", tab_id="distrib"),
                    dbc.Tab(label="Lampadaires", tab_id="lampes"),
                    dbc.Tab(label="Chronologie", tab_id="chrono"),
                    dbc.Tab(label="Qualité des données", tab_id="qualite"),
                    dbc.Tab(label="Bruit et valeurs aberrantes", tab_id="bruit"),
                ])),
                dbc.CardBody([
                    html.Div(id="tableau"),
                    html.Div(id="p-distrib"),
                    html.Div(dcc.Graph(id="lampes", config={"displayModeBar": False}), id="p-lampes"),
                    html.Div([html.Div([html.I(className="bi bi-cursor me-1"), "Cliquez une heure : la carte montre "
                                        "l'endroit où passait le véhicule et Street View s'y ouvre."],
                                       className="small text-muted mb-1"),
                              dcc.Graph(id="chrono", config={"displayModeBar": False})], id="p-chrono"),
                    html.Div(id="qualite"),
                    html.Div(id="bruit"),
                ]),
            ])),
        ]),
        html.Footer([html.A("Hackathon 2026 Bouygues Energies & Services", href=LIEN_HACKATHON, target="_blank",
                            rel="noopener"), " · équipe Clair-Obscur · mesures GETYS · lampadaires estimés et "
                     "palette de carte.html (getys_rapport.py) · filaire © contributeurs OpenStreetMap"],
                    className="pied"),
    ])),

    # --- envoi d'une nouvelle tournée
    dbc.Modal(id="fenetre-envoi", is_open=False, centered=True, size="lg", children=[
        dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-cloud-arrow-up me-2"),
                                        "Analyser une nouvelle tournée"])),
        dbc.ModalBody([
            html.H6("Mesures de la tournée", className="fw-semibold"),
            dcc.Upload(id="upload", accept=".tdms", style=upload_style, children=html.Div(
                [html.I(className="bi bi-file-earmark-binary fs-3 d-block mb-1 text-warning"),
                 "Glissez le fichier ", html.B(".tdms"), " ou ", html.A("parcourir")])),
            html.Div(id="tdms-nom", className="mt-2 small"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Annuler", id="btn-annuler", color="secondary", outline=True),
            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Lancer l'analyse"], id="btn-lancer",
                       color="warning", disabled=True, className="fw-semibold"),
        ]),
    ]),
])

SDAL_INPUTS = [Input(f"sdal-{c}", "value") for c in T.CLASSES] + [Input("sdal-rues", "data")]
SDAL_STATES = [State(f"sdal-{c}", "value") for c in T.CLASSES]


def rues_de(valeurs):
    """Seuils par rue (fichier SDAL importé) : dernière valeur de SDAL_INPUTS."""
    return (valeurs[len(T.CLASSES)] if len(valeurs) > len(T.CLASSES) else None) or {}


def sdal_de(valeurs):
    return {c: (v if v is not None else T.CLASSES[c][2]) for c, v in zip(T.CLASSES, valeurs)}


# ------------------------------------------------------------------ CALLBACKS : ENVOI
@app.callback(Output("fenetre-envoi", "is_open"),
              Input("btn-nouvelle", "n_clicks"), Input("btn-annuler", "n_clicks"),
              prevent_initial_call=True)
def ouvrir_fenetre(n_ouvrir, n_annuler):
    return ctx.triggered_id == "btn-nouvelle"


@app.callback(Output("btn-lancer", "disabled"), Output("tdms-nom", "children"),
              Input("upload", "filename"), prevent_initial_call=True)
def fichier_choisi(nom):
    if nom and nom.lower().endswith(".tdms"):
        return False, html.Span([html.I(className="bi bi-check-circle me-1"), nom], className="text-success")
    return True, html.Span("Seuls les fichiers .tdms sont acceptés.", className="text-warning")


@app.callback(Output("message", "children"), Output("choix-rapport", "options"),
              Output("choix-rapport", "value"), Output("fenetre-envoi", "is_open", allow_duplicate=True),
              Input("btn-lancer", "n_clicks"),
              State("upload", "contents"), State("upload", "filename"),
              State("choix-rapport", "value"), prevent_initial_call="initial_duplicate",
              running=[(Output("voile-analyse", "style"), {"display": "flex"}, {"display": "none"})])
def envoyer(n_lancer, contenu, nom, actuel):
    msg = None
    dossier_retour = None
    lance = ctx.triggered_id == "btn-lancer"
    if lance and contenu:
        fichiers = {"file": (nom, base64.b64decode(contenu.split(",")[1]))}
        try:
            rep = requests.post(SERVER_URL, files=fichiers, timeout=600)
            if rep.ok:
                msg = dbc.Alert([html.I(className="bi bi-check-circle me-2"),
                                 f"Analyse de {nom} terminée."], color="success", className="py-2",
                                dismissable=True, duration=8000)
                try:     # si le serveur renvoie {"dossier": "..."} on l'utilise
                    dossier_retour = rep.json().get("dossier")
                except ValueError:
                    pass
            else:
                try:
                    detail = rep.json().get("detail") or rep.json().get("erreur", "")
                except ValueError:
                    detail = ""
                msg = dbc.Alert([html.B(f"Erreur serveur : {rep.status_code}"),
                                 html.Pre(detail, className="small mb-0 mt-2")],
                                color="danger", className="py-2", dismissable=True)
        except requests.RequestException as e:
            msg = dbc.Alert(f"Serveur d'analyse injoignable ({SERVER_URL}) : {e}", color="danger",
                            className="py-2", dismissable=True)

    rapports = lister_rapports()
    options = [{"label": d.name.replace("_", " · "), "value": str(d)} for d in rapports]
    if dossier_retour:
        valeur = str(RAPPORTS_DIR / Path(dossier_retour).name)
    elif lance and rapports:
        valeur = str(rapports[0])            # le plus récent = celui qui vient d'être produit
    else:
        valeur = actuel or (str(rapports[0]) if rapports else None)
    return msg, options, valeur, False


@app.callback(Output("lien-carte", "href"), Output("lien-carte", "className"), Output("urls-prechargement", "data"),
              Input("choix-rapport", "value"))
def liens_rapport(dossier):
    """« Carte satellite » : la heatmap de heatmap_flux.py en plein écran sur fond satellite (même page que la vue
    Heatmap) ; URL des calques à précharger. Le rapport se compose dans la vue Chargé d'affaires."""
    classe = "btn btn-outline-light btn-sm"
    if not dossier:
        return None, classe + " disabled", None
    data = charger(dossier)
    carte = f"{url_heatmap(charger(dossier, brut=False))}#fond=satellite" if data["hex"] is not None else None
    return carte, classe if carte else classe + " disabled", [url_couche(data, "hexagones")]


# « Lancer l'analyse » : la fenêtre se ferme et l'écran d'analyse s'affiche tout de suite, dans le navigateur
# (sans attendre l'envoi du fichier au serveur) ; le callback d'envoi le retire à la fin (running)
app.clientside_callback(
    """function(n) { return n ? [false, {display: 'flex'}] : [window.dash_clientside.no_update,
                                                            window.dash_clientside.no_update]; }""",
    Output("fenetre-envoi", "is_open", allow_duplicate=True),
    Output("voile-analyse", "style", allow_duplicate=True),
    Input("btn-lancer", "n_clicks"), prevent_initial_call=True)


# voile « Chargement de la tournée… » : affiché dès qu'une tournée est choisie, retiré quand les indicateurs
# sont calculés ET que la carte a dessiné ses données (ou au bout de 15 s)
app.clientside_callback(
    """function(dossier) {
        // repère les données affichées AVANT le changement : le voile attend qu'elles soient remplacées
        const m = window.getysCarte; window.__avantChargement = null;
        if (m) m.eachLayer(l => { if (!window.__avantChargement && l.feature) window.__avantChargement = l.feature; });
        return dossier ? {display: 'flex'} : {display: 'none'};
    }""",
    Output("voile-chargement", "style"), Input("choix-rapport", "value"))
app.clientside_callback(
    """async function(kpis) {
        const t0 = Date.now();
        const pret = () => {                       // la carte montre de NOUVELLES données
            const m = window.getysCarte; let nouveau = false;
            if (m) m.eachLayer(l => { if (l.feature && l.feature !== window.__avantChargement) nouveau = true; });
            return nouveau;
        };
        while (!pret() && Date.now() - t0 < 15000) await new Promise(r => setTimeout(r, 100));
        return {display: 'none'};
    }""",
    Output("voile-chargement", "style", allow_duplicate=True), Input("kpis", "children"), prevent_initial_call=True)


# données des vues Hexagones / Points téléchargées en arrière-plan : elles s'ouvrent sans attente
app.clientside_callback(
    """function(urls) {
        (urls || []).forEach((u, i) => setTimeout(() => fetch(u).catch(() => null), 1500 + 1000 * i));
        return '';
    }""",
    Output("prechargement", "children"), Input("urls-prechargement", "data"))


ONGLETS = {"troncons": "tableau", "distrib": "p-distrib", "lampes": "p-lampes", "chrono": "p-chrono",
           "qualite": "qualite", "bruit": "bruit"}


@app.callback(*[Output(p, "style") for p in ONGLETS.values()], Input("onglets", "active_tab"))
def changer_onglet(onglet):
    return [{"display": "block" if o == onglet else "none"} for o in ONGLETS]


# ------------------------------------------------------------------ CALLBACKS : AFFICHAGE
@app.callback(Output("contexte", "children"), Output("kpis", "children"),
              Output("p-distrib", "children"), Output("lampes", "figure"), Output("chrono", "figure"),
              Output("tableau", "children"), Output("qualite", "children"), Output("bruit", "children"),
              Input("choix-rapport", "value"), *SDAL_INPUTS)
def afficher(dossier, *valeurs):
    if not dossier:
        vide = carte_vide("Aucune tournée analysée — cliquez sur « Nouvelle tournée »", 360)
        return None, [], None, vide, vide, None, None, None
    sdal = sdal_de(valeurs)
    data = charger(dossier)
    meta = data["resume"].get("meta", {})
    k = data["resume"].get("kpi", {})
    tr = avec_seuils(data["troncons"], sdal, rues_de(valeurs)) if data["troncons"] is not None else None

    contexte = html.Div([
        html.Span([html.I(className="bi bi-geo-alt me-1"), meta.get("ville", Path(dossier).name)]),
        html.Span([html.I(className="bi bi-calendar3 me-1"), meta.get("date", "–")]),
        html.Span([html.I(className="bi bi-clock me-1"), f"{meta.get('debut', '?')} → {meta.get('fin', '?')}"])
        if meta.get("debut") else None,
        html.Span([html.I(className="bi bi-signpost-split me-1"), f"{fmt(k.get('dist_km'), ' km')} parcourus"])
        if k.get("dist_km") else None,
        html.Span([html.I(className="bi bi-lightbulb me-1"),
                   f"{fmt(k.get('n_lampadaires'), dec=0)} lampadaires estimés"]) if k.get("n_lampadaires") else None,
    ], className="d-flex flex-wrap gap-3")
    alerte = None
    if data["erreur_troncons"]:
        alerte = dbc.Alert(["Moyenne au tronçon indisponible : filaire OpenStreetMap injoignable. ",
                            html.Small(data["erreur_troncons"])], color="warning", className="py-2 mt-2")

    coul = lambda t: "success" if t >= 80 else "warning" if t >= 50 else "danger"   # noqa: E731
    kpis = []
    if tr is not None and not tr.empty:
        lg = tr["longueur_m"]
        part_lin = float(100 * lg[tr["conforme"]].sum() / lg.sum())
        n_conf = int(tr["conforme"].sum())
        kpis += [
            kpi("rulers", "Linéaire conforme", fmt(part_lin, " %", 0),
                f"{fmt(lg[tr['conforme']].sum() / 1000, ' km')} sur {fmt(lg.sum() / 1000, ' km')}",
                coul(part_lin), barre=part_lin),
            kpi("bezier2", "Tronçons conformes", f"{n_conf} / {len(tr)}",
                f"{len(tr) - n_conf} sous leur seuil SDAL", coul(100 * n_conf / len(tr)),
                barre=100 * n_conf / len(tr)),
        ]
    kpis.append(kpi("speedometer2", "Distance mesurée", fmt(k.get("dist_km"), " km"),
                    f"en {fmt(k.get('duree_h'), ' h')}" if k.get("duree_h") else None, "info"))
    if k.get("n_lampadaires"):
        kpis.append(kpi("lightbulb", "Lampadaires estimés", fmt(k["n_lampadaires"], dec=0),
                        f"{fmt(k.get('puissance_tot_kw'), ' kW')} installés", "info"))

    return ([contexte, alerte], kpis, vue_types(tr), types_lampes(data),
            chronologie(data), tableau_troncons(tr), qualite(data, tr), bruit(data))


# ------------------------------------------------------------------ CALLBACKS : CARTE (Leaflet)
# Chaque calque charge ses données par URL (cache navigateur). Le mode affiche un calque et vide les autres ;
# la couleur et les seuils SDAL ne changent que le « hideout » des tronçons (recoloriage dans le navigateur).
# Leaflet garde sa vue : rien ne bouge quand on change de filtre.
@app.callback(Output("geo-troncons", "url"), Output("geo-troncons", "data"),
              Output("geo-hex", "url"), Output("geo-hex", "data"),
              Output("bandeau-lissage", "children"),
              Input("choix-rapport", "value"), Input("mode-carte", "value"), Input("lissage", "value"))
def afficher_calques(dossier, mode, lissage):
    if not dossier:
        return None, VIDE, None, VIDE, None
    brut = lissage == "avant" and mode in ("troncons", "hex")   # avant / après : tronçons et hexagones
    data = charger(dossier, brut=brut)
    sortie = []
    for m, couche in (("troncons", "troncons"), ("hex", "hexagones")):
        sortie += [url_couche(data, couche), None] if mode == m else [None, VIDE]
    bandeau = []                                   # messages d'erreur uniquement
    if mode == "troncons" and data["troncons"] is None:
        bandeau.append(html.Div("Filaire de voirie indisponible (réseau ?) : " + (data["erreur_troncons"] or ""),
                                className="bandeau-lissage"))
    if mode == "heatmap":
        bandeau.append(html.Div([html.I(className="bi bi-thermometer-half me-2"),
                                 "Heatmap : moyenne des deux sens, lissée (σ = 4 m), et lampadaires estimés. Vue lissée / "
                                 "3 capteurs, lampadaires et opacité se règlent dans le panneau de la carte. Premier "
                                 "affichage d'une tournée : quelques secondes."], className="bandeau-lissage"))
    if brut and not data["brut"]:
        bandeau.append(html.Div([html.I(className="bi bi-soundwave me-2"), "Ce rapport ne contient pas le signal "
                                 "brut (colonne E_brut) : relancez l'analyse pour l'obtenir."], className="bandeau-lissage"))
    return (*sortie, bandeau or None)


@app.callback(Output("geo-troncons", "hideout"), Input("couleur-carte", "value"), Input("mode-carte", "value"),
              *SDAL_INPUTS)
def colorer_troncons(couleur, mode, *valeurs):
    """Mode de couleur et seuils SDAL : appliqués par la fonction de style dans le navigateur."""
    return {"couleur": couleur if mode == "troncons" else "eclairement", "sdal": sdal_de(valeurs), "rues": rues_de(valeurs)}


@app.callback(Output("groupe-aberrants", "children"),
              Input("couches", "value"), Input("choix-rapport", "value"), Input("mode-carte", "value"))
def afficher_couches(couches, dossier, mode):
    """Valeurs aberrantes retirées par le filtrage : par-dessus les vues Tronçons et Hexagones."""
    if not dossier or mode not in ("troncons", "hex") or "ab" not in (couches or []):
        return []
    return marqueurs_aberrants(charger(dossier)["aberrants"])


@app.callback(Output("carte-heatmap", "src"), Output("carte-heatmap", "style"),
              Input("choix-rapport", "value"), Input("mode-carte", "value"), Input("fond", "value"))
def afficher_heatmap(dossier, mode, fond):
    """Vue Heatmap : page de heatmap_flux.py posée sur la carte. Seul le fond change -> seul le #fond= change,
    la page n'est pas rechargée. Page générée au premier affichage seulement."""
    if not dossier or mode != "heatmap":
        return dash.no_update, {"display": "none"}
    return f"{url_heatmap(charger(dossier))}#fond={fond}&integre=1", {"display": "block"}


@app.callback(Output("export-zones", "href"), Output("export-cellules", "href"), Input("choix-rapport", "value"))
def liens_export(dossier):
    """Exports GeoJSON de la heatmap (mêmes fichiers que heatmap_flux.py), téléchargés depuis la barre d'outils."""
    if not dossier:
        return None, None
    base = f"/heatmap/{quote(Path(dossier).name)}"
    return f"{base}/heatmap_flux.geojson", f"{base}/heatmap_flux_points.geojson"


@app.callback(Output("outil-export", "style"), Input("mode-carte", "value"))
def montrer_export(mode):
    return {} if mode == "heatmap" else {"display": "none"}


@app.callback(Output("outil-couleur-carte", "style"), Output("outil-lissage", "style"), Output("couches", "style"),
              Input("mode-carte", "value"))
def montrer_outils(mode):
    """Couleur (conformité SDAL) : Tronçons. Avant / après lissage et valeurs aberrantes : Tronçons et Hexagones."""
    cache = {"display": "none"}
    deux = {} if mode in ("troncons", "hex") else cache
    return ({} if mode == "troncons" else cache), deux, deux


@app.callback(Output("legende", "children"), Input("mode-carte", "value"), Input("couleur-carte", "value"),
              Input("couches", "value"), Input("choix-rapport", "value"))
def afficher_legende(mode, couleur, couches, dossier):
    return legende(mode, couleur if mode == "troncons" else "eclairement",
                   couches if mode in ("troncons", "hex") else [], charger(dossier) if dossier else None)


# thème clair / sombre : bouton de l'en-tête, mémorisé dans le navigateur (dcc.Store local)
app.clientside_callback(
    """function(n, theme) { return n ? (theme === 'sombre' ? 'clair' : 'sombre') : window.dash_clientside.no_update; }""",
    Output("theme", "data"), Input("btn-theme", "n_clicks"), State("theme", "data"), prevent_initial_call=True)
app.clientside_callback(
    """function(theme) {
        const sombre = theme === 'sombre', h = document.documentElement;
        h.setAttribute('data-theme', sombre ? 'sombre' : 'clair');
        h.setAttribute('data-bs-theme', sombre ? 'dark' : 'light');
        return [sombre ? 'bi bi-sun' : 'bi bi-moon-stars', sombre ? 'Mode clair' : 'Mode sombre', ''];
    }""",
    Output("icone-theme", "className"), Output("btn-theme", "title"), Output("theme-applique", "children"),
    Input("theme", "data"))


@app.callback(Output("fond", "value"), Output("op-fond", "value"),
              Input("theme", "data"), Input("fond", "value"), Input("op-fond", "value"))
def synchro_fond(theme, fond, fond_op):
    """Un seul fond pour les deux vues : le choix fait dans l'une vaut pour l'autre. Il suit le thème (sauf
    satellite, choisi exprès)."""
    if ctx.triggered_id == "op-fond":
        v = fond_op
    elif ctx.triggered_id == "fond":
        v = fond
    else:
        v = fond if fond == "satellite" else ("sombre" if theme == "sombre" else "clair")
    return v, v


app.clientside_callback(
    ClientsideFunction(namespace="getys", function_name="appliquerFond"),
    Output("fond-vectoriel", "children"), Input("fond", "value"))


@app.callback(Output("fonds", "children"), Output("cadre-carte", "className"), Input("fond", "value"))
def changer_fond(fond):
    # la classe va sur le bloc parent : dash-leaflet ne met pas à jour className de la carte après sa création
    return tuiles(fond), f"carte-leaflet fond-{fond}"


@app.callback(Output("carte", "viewport"), Input("choix-rapport", "value"))
def cadrer_tournee(dossier):
    """Nouvelle tournée : vue d'ensemble sur les mesures."""
    if not dossier:
        return dash.no_update
    m = charger(dossier)["mesures"]
    return {"center": [float(m["lat"].mean()), float(m["lon"].mean())], "zoom": 14.5}


@app.callback(Output("couleur-carte", "options"), Output("couleur-carte", "value"),
              Input("mode-carte", "value"), State("couleur-carte", "value"))
def options_couleur(mode, valeur):
    """La conformité SDAL ne concerne que les tronçons : option grisée pour les autres vues."""
    tr = mode == "troncons"
    options = [{"label": "Éclairement", "value": "eclairement"},
               {"label": "Conformité SDAL", "value": "conformite", "disabled": not tr}]
    return options, valeur if tr else "eclairement"


# ------------------------------------------------------------------ CALLBACKS : SÉLECTION -> CARTE + STREET VIEW
def _selection_troncon(data, tid, titre_extra=None):
    r = data["troncons"].set_index("id").loc[tid]
    lat, lon, cap, _ = point_le_plus_sombre(data, tid)
    glon, glat = data["geoms"][tid]
    detail = (f"point le plus sombre · tronçon {fmt(r['E_moy'], ' lx')} "
              f"(sens 1 : {fmt(r['E_sens1'], ' lx')}, sens 2 : {fmt(r['E_sens2'], ' lx')})")
    if titre_extra:
        detail = f"{titre_extra} · {detail}"
    lg = float(r["longueur_m"])
    return {"lat": lat, "lon": lon, "cap": cap, "titre": r["nom"], "detail": detail, "centrer": True,
            "tid": tid, "tr_lat": list(glat), "tr_lon": list(glon),
            "c_lat": float(np.mean(glat)), "c_lon": float(np.mean(glon)),
            "zoom": 18.0 if lg < 80 else 17.5 if lg < 180 else 17.0}


@app.callback(Output("selection", "data"),
              Input("geo-troncons", "clickData"), Input("geo-hex", "clickData"),
              Input("table-troncons", "active_cell"), Input("table-qualite", "active_cell"),
              Input("table-aberrants", "active_cell"), Input({"type": "aberrant", "k": ALL}, "n_clicks"),
              Input("chrono", "clickData"),
              State("choix-rapport", "value"), prevent_initial_call=True)
def choisir_point(clic_tr, clic_hex, cellule, cellule_q, cellule_ab, clics_ab, clic_chrono, dossier):
    if not dossier:
        return dash.no_update
    data = charger(dossier)
    src = ctx.triggered_id
    if src == "chrono":                                  # chronologie : lieu du véhicule à l'heure cliquée
        m = data["mesures"]
        if not clic_chrono or m is None or "heure_locale" not in m:
            return dash.no_update
        h = float(clic_chrono["points"][0]["x"])
        i = int(np.argmin(np.abs(((m["heure_locale"].to_numpy(float) - h + 12) % 24) - 12)))
        p = m.iloc[i]
        hh = float(p["heure_locale"])
        return {"lat": float(p["lat"]), "lon": float(p["lon"]), "cap": float(p.get("cap", 0) or 0), "centrer": True,
                "zoom": 17.5, "titre": f"Chronologie — {int(hh)}h{int(round(hh % 1 * 60)) % 60:02d}",
                "detail": f"{fmt(p['lux'], ' lx')} mesurés à cet instant"}
    # valeur aberrante retirée : ligne de l'onglet « Bruit et valeurs aberrantes » ou cercle de la carte
    k_ab = None
    if src == "table-aberrants" and cellule_ab:
        k_ab = cellule_ab.get("row_id")
    elif isinstance(src, dict) and src.get("type") == "aberrant" and ctx.triggered and ctx.triggered[0]["value"]:
        k_ab = src["k"]
    if src == "table-aberrants" or isinstance(src, dict):
        ab = data["aberrants"]
        if k_ab is None or ab is None or k_ab not in set(ab["id"]):
            return dash.no_update
        a = ab.set_index("id").loc[k_ab]
        if pd.isna(a["lat"]):
            return dash.no_update
        lat, lon = float(a["lat"]), float(a["lon"])
        return {"lat": lat, "lon": lon, "cap": cap_proche(data, lat, lon), "centrer": src == "table-aberrants",
                "titre": f"⚠ Valeur aberrante retirée — {a['motif']}",
                "detail": f"pic {fmt(a['pic_lx'], ' lx', 0)} (fond {fmt(a['fond_lx'], ' lx', 0)}) · largeur "
                          f"{fmt(a['largeur_m'], ' m')} · {a['heure']} · position {a['precision']}"}
    clics = {"geo-troncons": clic_tr, "geo-hex": clic_hex}
    clic = clics.get(src) if isinstance(src, str) else None
    if clic:
        p, geo = clic.get("properties") or {}, clic.get("geometry") or {}
        if p.get("_lat") is not None:                         # point exact du clic (mémorisé côté navigateur)
            lat, lon = p["_lat"], p["_lon"]
        elif geo.get("type") == "Point":
            lon, lat = geo["coordinates"]
        elif geo.get("type") == "Polygon":                    # hexagone : centre de la cellule
            ring = geo["coordinates"][0][:-1]
            lon, lat = float(np.mean([c[0] for c in ring])), float(np.mean([c[1] for c in ring]))
        else:
            return dash.no_update
        sel = {"lat": lat, "lon": lon, "cap": cap_proche(data, lat, lon), "titre": "Point sélectionné",
               "detail": None, "centrer": False}
        if src == "geo-troncons" and data["geoms"] and p.get("id") in data["geoms"]:
            glon, glat = data["geoms"][p["id"]]
            sens = (f"sens unique ({p['np']} passage(s))" if p.get("su") else
                    f"sens 1 : {fmt(p.get('s1'), ' lx')} · sens 2 : {fmt(p.get('s2'), ' lx')}")
            sel.update(titre=p["n"], detail=f"{fmt(p.get('E'), ' lx')} · {sens}", tid=p["id"],
                       tr_lat=list(glat), tr_lon=list(glon))
        elif src == "geo-hex":
            sel.update(titre="Hexagone de 2 m", detail=f"{fmt(p.get('E'), ' lx')} (sens 1 : {fmt(p.get('s1'), ' lx')}, "
                                                       f"sens 2 : {fmt(p.get('s2'), ' lx')}) · {p.get('np')} passage(s)")
        return sel
    tid, extra = None, None
    if src == "table-troncons" and cellule:
        tid = cellule.get("row_id")
    elif src == "table-qualite" and cellule_q and cellule_q.get("row_id"):
        tid = cellule_q["row_id"].split("|")[0]
        extra = "à fiabiliser"
    if tid is None or data["geoms"] is None or tid not in data["geoms"]:
        return dash.no_update
    return _selection_troncon(data, tid, extra)


@app.callback(Output("selection", "data", allow_duplicate=True), Input("choix-rapport", "value"),
              prevent_initial_call=True)
def oublier_selection(dossier):
    """Nouvelle tournée : le point sélectionné (carte, Street View) appartenait à l'ancienne."""
    return None


def streetview_vide():
    return html.Div([html.I(className="bi bi-geo-alt fs-2 d-block mb-2"),
                     "Sélectionnez un point ou un tronçon pour voir la rue."], className="sv-vide")


@app.callback(Output("sv-cadre", "children"), Output("sv-legende", "children"),
              Output("sv-externe", "href"), Output("sv-externe", "className"),
              Input("selection", "data"), prevent_initial_call=True)
def afficher_streetview(sel):
    if not sel:
        return streetview_vide(), None, None, "small text-info d-none"
    cadre = html.Iframe(src=embed_streetview(sel["lat"], sel["lon"], sel.get("cap", 0)), className="sv-iframe",
                        allow="fullscreen")
    legende = [html.Div(sel.get("titre"), className="fw-semibold"),
               html.Div(sel.get("detail") or "", className="small text-muted"),
               html.Div(f"{sel['lat']:.5f}, {sel['lon']:.5f}", className="small text-muted")]
    return cadre, legende, lien_streetview(sel["lat"], sel["lon"], sel.get("cap", 0)), "small text-info"


@app.callback(Output("groupe-selection", "children"), Output("carte", "viewport", allow_duplicate=True),
              Input("selection", "data"), prevent_initial_call=True)
def marquer_selection(sel):
    """Surligne le tronçon (ou la rue), place le point Street View et, depuis une liste, y emmène la carte."""
    if not sel:
        return [], dash.no_update
    lignes, ligne = [], []
    for la, lo in zip(sel.get("tr_lat") or [], sel.get("tr_lon") or []):
        if la is None:
            if ligne:
                lignes.append(ligne)
            ligne = []
        else:
            ligne.append([la, lo])
    if ligne:
        lignes.append(ligne)
    calques = [dl.Polyline(positions=l, color=CYAN, weight=10, opacity=0.55, interactive=False) for l in lignes]
    calques.append(dl.CircleMarker(center=[sel["lat"], sel["lon"]], radius=9, color="#0b0e13", weight=2,
                                   fillColor=CYAN, fillOpacity=0.95, interactive=False))
    if not sel.get("centrer"):
        return calques, dash.no_update
    return calques, {"center": [sel.get("c_lat", sel["lat"]), sel.get("c_lon", sel["lon"])],
                     "zoom": sel.get("zoom", 17.5), "transition": "setView"}   # direct : flyTo dézoome puis rezoome


# sélection depuis une liste : on fait défiler la page jusqu'à la carte
app.clientside_callback(
    """function(sel) {
        if (sel && sel.centrer) {
            const el = document.getElementById('bloc-carte');
            if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
        }
        return '';
    }""",
    Output("defilement", "children"), Input("selection", "data"), prevent_initial_call=True)


# ------------------------------------------------------------------ CALLBACKS : VUES PAR RÔLE
PAGES = ("operateur", "analyste", "affaires", "doc")


# changement de vue dans le navigateur (instantané) ; une carte Leaflet créée dans une page masquée a une taille
# nulle : une fois la page affichée, on lui redonne sa taille sans déplacer la vue (opérateur : cadrage de la trace)
app.clientside_callback(
    """function(page) {
        page = page || 'analyste';
        const pages = ['operateur', 'analyste', 'affaires', 'doc'];
        const redimensionner = (essai) => {
            const m = page === 'operateur' ? window.getysCarteOperateur : page === 'analyste' ? window.getysCarte : null;
            if (!m) return;
            const el = m.getContainer();
            if (!el.offsetWidth && essai < 20) return setTimeout(() => redimensionner(essai + 1), 50);
            const c = m.getCenter(), z = m.getZoom();
            m.invalidateSize(false);
            m.setView(c, z, {animate: false});
        };
        setTimeout(() => redimensionner(0), 30);
        return pages.map(p => ({display: p === page ? 'block' : 'none'}));
    }""",
    *[Output(f"page-{p}", "style") for p in PAGES], Input("page", "value"))


# ------------------------------------------------------------------ CALLBACKS : SDAL (fichier)
@app.callback(Output("sdal-rues", "data"), Output("sdal-message", "children"),
              *[Output(f"sdal-{c}", "value") for c in T.CLASSES],
              Input("sdal-upload", "contents"), Input({"type": "sdal-retirer", "k": ALL}, "n_clicks"),
              State("sdal-upload", "filename"), State("sdal-rues", "data"), prevent_initial_call=True)
def importer_sdal(contenu, retirer, nom, rues):
    rien = [dash.no_update] * len(T.CLASSES)
    if isinstance(ctx.triggered_id, dict):                   # « Retirer les seuils par rue »
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return dash.no_update, dash.no_update, *rien
        return {}, html.Span("Seuils par rue retirés.", className="text-muted"), *rien
    if not contenu:
        return dash.no_update, dash.no_update, *rien
    try:
        res = SF.lire(contenu, T.CLASSES)
    except (ValueError, UnicodeDecodeError) as ex:
        return dash.no_update, html.Span(f"{nom} : {ex}", className="text-danger"), *rien
    if res["rues"]:
        msg = html.Span([html.I(className="bi bi-check-circle me-1"), f"{nom} : {res['message']} ",
                         html.A("Retirer", id={"type": "sdal-retirer", "k": 0}, href="#", className="ms-1")],
                        className="text-success")
        return res["rues"], msg, *rien
    valeurs = [res["types"].get(c, dash.no_update) for c in T.CLASSES]
    return dash.no_update, html.Span([html.I(className="bi bi-check-circle me-1"), f"{nom} : {res['message']}"],
                                     className="text-success"), *valeurs


# ------------------------------------------------------------------ CALLBACKS : OPÉRATEUR
def tdms_du_rapport(dossier):
    """Nom du .tdms dont est issu un rapport (resume.json -> meta.fichier_tdms)."""
    if not dossier or not (Path(dossier) / "resume.json").exists():
        return None
    nom = charger(dossier)["resume"].get("meta", {}).get("fichier_tdms")
    return Path(nom).name if nom else None


@app.callback(Output("op-fichier", "data"), Output("op-nom", "children"), Output("op-etat", "children"),
              Input("choix-rapport", "value"))
def fichier_operateur(dossier):
    """Le .tdms de la vue Opérateur est celui de la tournée choisie en haut (liste ou « Nouvelle tournée »).
    Jamais de repli sur un autre fichier : s'il manque, la vue le dit et reste vide."""
    voulu = tdms_du_rapport(dossier)
    if voulu and OP.chemin_tdms(voulu) is not None:
        return voulu, voulu, None
    msg = dbc.Alert(f"Le fichier {voulu} de la tournée choisie n'est pas disponible (tdms_recus/) : rechargez-le avec "
                    "« Nouvelle tournée » pour voir sa trace brute.", color="warning", className="m-2 py-2") \
        if voulu else None
    return None, "–", msg


@app.callback(Output("op-trace-url", "data"), Output("op-controles", "children"),
              Output("op-carte", "viewport"), Output("op-notes", "value"), Input("op-fichier", "data"))
def afficher_brut(nom):
    if not nom:
        return None, html.P("Aucun fichier .tdms pour cette tournée.",
                            className="text-muted small"), dash.no_update, ""
    f = OP.chemin_tdms(nom)
    try:
        brut = OP.lire_brut(f)
    except Exception as ex:                                   # fichier incomplet ou d'un autre format
        return None, dbc.Alert(f"Lecture impossible : {ex}", color="danger", className="py-2"), dash.no_update, ""
    d = brut["points"]
    lignes = [html.Div(className="controle", children=[
        html.I(className="bi bi-check-circle-fill text-success" if c["ok"] else "bi bi-exclamation-triangle-fill text-warning"),
        html.Div([html.Div(c["titre"], className="fw-semibold"), html.Div(c["detail"], className="small text-muted")])])
        for c in brut["controles"]]
    lignes.append(html.Div(f"Lecture du fichier : {brut['duree_lecture']} s · {len(d)} positions GPS × 3 capteurs",
                           className="small text-muted mt-2"))
    lat0 = float(d["lat"].mean())                            # zoom qui fait tenir la trace (carte ≈ 900 × 640 px)
    dx = (d["lon"].max() - d["lon"].min()) * 111320 * math.cos(math.radians(lat0))
    dy = (d["lat"].max() - d["lat"].min()) * 110540
    mpp = max(dx / 900, dy / CARTE_H, 0.5) * 1.15
    zoom = int(np.clip(math.floor(math.log2(40075016.686 * math.cos(math.radians(lat0)) / (mpp * 256))), 10, 18))
    vue = {"center": [(d["lat"].min() + d["lat"].max()) / 2, (d["lon"].min() + d["lon"].max()) / 2], "zoom": zoom}
    return f"/operateur/{quote(Path(nom).stem)}.json?v={int(f.stat().st_mtime)}", lignes, vue, \
        OP.lire_notes(nom).get("notes", "")


# la trace (3 × 60 000 points) est dessinée par le navigateur sur un canevas (assets/carte_leaflet.js)
app.clientside_callback(
    """function(url) { window.getys.traceBrute(url); return ''; }""",
    Output("op-trace-dessin", "children"), Input("op-trace-url", "data"))


@app.callback(Output("op-fonds", "children"), Output("op-cadre", "className"), Input("fond", "value"))
def fond_operateur(fond):
    """Mêmes fonds que la carte de l'analyste : vectoriel en clair / sombre (assets/carte_leaflet.js, nets à tous
    les zooms), images Esri haute résolution en satellite."""
    return tuiles(fond, "op-tuiles"), f"carte-leaflet fond-{fond}"


@app.callback(Output("op-form", "style"), Output("op-coord", "children"), Output("op-attente", "data"),
              Input("op-carte", "clickData"), State("op-mode-note", "value"), prevent_initial_call=True)
def placer_point(clic, mode):
    """Mode « Ajouter un point au clic » : le clic sur la carte prépare un point à commenter."""
    if "on" not in (mode or []) or not clic or "latlng" not in clic:
        return dash.no_update, dash.no_update, dash.no_update
    lat, lon = clic["latlng"]["lat"], clic["latlng"]["lng"]
    return {"display": "block"}, f"Nouveau point : {lat:.5f}, {lon:.5f}", [lat, lon]


def _marqueurs(notes, attente=None):
    # volet des marqueurs (z 600) : au-dessus du canevas des lux bruts (volet traceBrute, z 450)
    calques = [dl.CircleMarker(id={"type": "op-point", "k": i - 1}, center=[p["lat"], p["lon"]], radius=9,
                               color="#fff", weight=2, fillColor="#D63AF9", fillOpacity=0.95, pane="markerPane",
                               bubblingMouseEvents=False, children=dl.Tooltip(f"{i}. {p.get('texte', '')}"))
               for i, p in enumerate(notes.get("points", []), 1)]
    if attente:
        calques.append(dl.CircleMarker(center=attente, radius=9, color="#111", weight=1.5, dashArray="3",
                                       fillColor="#ffffff", fillOpacity=0.8, pane="markerPane"))
    return calques


def _liste(notes):
    if not notes.get("points"):
        return html.Div("Aucun point pour le moment.", className="small text-muted mb-1")
    return html.Div(className="liste-points", children=[
        html.Div(className="point-note", children=[
            html.Span(f"{i}", className="deficit-rang"),
            html.Div([html.Div(p.get("texte") or "(sans commentaire)"),
                      html.Div(f"{p['lat']:.5f}, {p['lon']:.5f} · {p.get('date', '')}", className="small text-muted")],
                     id={"type": "op-aller", "k": i - 1}, className="flex-grow-1 point-aller",
                     title="Voir ce point sur la carte"),
            html.Button(html.I(className="bi bi-trash"), id={"type": "op-suppr", "k": i - 1},
                        className="btn btn-sm btn-link text-danger p-0", title="Supprimer")])
        for i, p in enumerate(notes["points"], 1)])


# clic sur un point annoté (liste ou carte) : la carte se centre dessus et ouvre son commentaire
app.clientside_callback(
    """function(liste, carte, marqueurs) {
        const t = dash_clientside.callback_context.triggered;
        if (!t.length || !t[0].value) return dash_clientside.no_update;
        const k = JSON.parse(t[0].prop_id.split('.')[0]).k, m = window.getysCarteOperateur;
        const p = (marqueurs || [])[k];
        if (!m || !p) return dash_clientside.no_update;
        const centre = p.props.center, r = m.getContainer().getBoundingClientRect();
        if (r.top < 0 || r.bottom > window.innerHeight)              // liste sous la carte (écran étroit)
            m.getContainer().scrollIntoView({behavior: 'smooth', block: 'center'});
        // directement sur le point (flyTo dézoome puis rezoome en survolant la carte : déroutant)
        m.setView(centre, Math.max(m.getZoom(), 19), {animate: false});
        m.eachLayer(l => {
            if (l.getTooltip && l.getTooltip() && l.getLatLng && l.getLatLng().equals(L.latLng(centre))) l.openTooltip();
        });
        return '';
    }""",
    Output("op-aller-fait", "children"), Input({"type": "op-aller", "k": ALL}, "n_clicks"),
    Input({"type": "op-point", "k": ALL}, "n_clicks"), State("op-marqueurs", "children"), prevent_initial_call=True)


@app.callback(Output("op-marqueurs", "children"), Output("op-liste", "children"), Output("op-texte", "value"),
              Output("op-form", "style", allow_duplicate=True),
              Input("op-fichier", "data"), Input("op-ajouter", "n_clicks"), Input({"type": "op-suppr", "k": ALL}, "n_clicks"),
              Input("op-attente", "data"),
              State("op-texte", "value"), prevent_initial_call="initial_duplicate")
def points_operateur(nom, ajouter, suppr, attente, texte):
    if not nom:
        return [], _liste({}), dash.no_update, dash.no_update
    notes = OP.lire_notes(nom)
    src = ctx.triggered_id
    if src == "op-ajouter" and ajouter and attente:
        notes["points"].append({"lat": attente[0], "lon": attente[1], "texte": (texte or "").strip(),
                                "date": pd.Timestamp.now(tz="Europe/Paris").strftime("%d/%m/%Y %H:%M")})
        OP.ecrire_notes(nom, notes)
        return _marqueurs(notes), _liste(notes), "", {"display": "none"}
    if isinstance(src, dict) and ctx.triggered and ctx.triggered[0]["value"]:
        if 0 <= src["k"] < len(notes["points"]):
            notes["points"].pop(src["k"])
            OP.ecrire_notes(nom, notes)
        return _marqueurs(notes), _liste(notes), dash.no_update, dash.no_update
    if src == "op-attente":                                   # point en attente de commentaire
        return _marqueurs(notes, attente), _liste(notes), dash.no_update, dash.no_update
    return _marqueurs(notes), _liste(notes), dash.no_update, {"display": "none"}


@app.callback(Output("op-notes-etat", "children"), Input("op-enregistrer", "n_clicks"),
              State("op-fichier", "data"), State("op-notes", "value"), prevent_initial_call=True)
def enregistrer_notes(n, nom, texte):
    if not nom:
        return "Aucun fichier pour cette tournée."
    notes = OP.lire_notes(nom)
    notes["notes"] = texte or ""
    OP.ecrire_notes(nom, notes)
    return f"Enregistré à {pd.Timestamp.now(tz='Europe/Paris').strftime('%H:%M')}."


# ------------------------------------------------------------------ CALLBACKS : CHARGÉ D'AFFAIRES
@app.callback(Output("ra-sections", "value"), Input("ra-defaut", "n_clicks"), prevent_initial_call=True)
def sections_par_defaut(n):
    return RA.DEFAUT


def _rapport_html(dossier, sections, titre, destinataire, auteur, commentaire, valeurs):
    data = charger(dossier)
    tr = avec_seuils(data["troncons"], sdal_de(valeurs), rues_de(valeurs)) if data["troncons"] is not None else None
    notes = OP.notes_du_rapport(data["resume"].get("meta", {}).get("fichier_tdms"))
    figures = {"lampes": types_lampes(data) if data["resume"].get("types") else None,
               "chrono": chronologie(data) if data["resume"].get("ts") else None}
    return RA.construire(sections or [], {"titre": titre, "destinataire": destinataire, "auteur": auteur,
                                          "commentaire": commentaire}, data, tr, T.CLASSES, figures, notes)


RA_STATES = [State("ra-sections", "value"), State("ra-titre", "value"), State("ra-destinataire", "value"),
             State("ra-auteur", "value"), State("ra-commentaire", "value"),
             *[State(f"sdal-{c}", "value") for c in T.CLASSES], State("sdal-rues", "data")]


@app.callback(Output("ra-cadre", "srcDoc"), Input("ra-apercu", "n_clicks"), Input("page", "value"),
              Input("choix-rapport", "value"), *RA_STATES)
def apercu_rapport(n, page, dossier, sections, titre, destinataire, auteur, commentaire, *valeurs):
    if page != "affaires" or not dossier:
        return dash.no_update
    return _rapport_html(dossier, sections, titre, destinataire, auteur, commentaire, valeurs)


@app.callback(Output("ra-download", "data"), Input("ra-telecharger", "n_clicks"), State("choix-rapport", "value"),
              *RA_STATES, prevent_initial_call=True)
def telecharger_rapport(n, dossier, sections, titre, destinataire, auteur, commentaire, *valeurs):
    if not dossier:
        return dash.no_update
    page = _rapport_html(dossier, sections, titre, destinataire, auteur, commentaire, valeurs)
    return dict(content=page, filename=f"rapport_{Path(dossier).name}.html", type="text/html")


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8050")),
            debug=os.environ.get("DEBUG", "1") == "1")
