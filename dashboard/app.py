"""Tableau de bord éclairage public — upload TDMS -> serveur -> affichage des sorties.

Trois lectures des mesures (cf. cahier des charges GETYS) :
  * Moyenne au tronçon : filaire OSM, une moyenne par sens de circulation puis moyenne des deux sens
  * Hexagones de 2 m   : bande continue le long du trajet (3 luxmètres décalés latéralement, comme
    getys_spatialisation.py), multi-passages, même règle par sens
  * Points chauds / froids : valeur moyenne autour de chaque point (méthode de carte.html, getys_rapport.py),
    rayon fixe en mètres quel que soit le zoom
Carte : Leaflet (dash-leaflet), comme carte.html. Les données de chaque calque sont servies à part (gzip, cache
navigateur) ; couleurs et seuils SDAL sont appliqués dans le navigateur (assets/carte_leaflet.js).
La conformité (SDAL : un seuil d'éclairement moyen par type de voie) ne s'applique qu'aux tronçons.
Chaque vue peut être affichée avant ou après le lissage du bruit (colonne E_brut de mesures_pas2m.csv).
"""
import base64
import gzip
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, unquote

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

import hexagones as H
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
FONDS = {   # (url, attribution, zoom natif max) — les tuiles raster CARTO demandent désormais une clé
    "sombre": [(ESRI.format("Canvas/World_Dark_Gray_Base"), ATTR_ESRI, 16),
               (ESRI.format("Canvas/World_Dark_Gray_Reference"), "", 16)],
    "clair": [(ESRI.format("Canvas/World_Light_Gray_Base"), ATTR_ESRI, 16),
              (ESRI.format("Canvas/World_Light_Gray_Reference"), "", 16)],
    "satellite": [(ESRI.format("World_Imagery"), "Imagerie © Esri, Maxar, Earthstar Geographics", 19),
                  (ESRI.format("Reference/World_Boundaries_and_Places"), "", 19)],
}


def tuiles(fond):
    """Au-delà du zoom natif, Leaflet agrandit les dernières tuiles au lieu d'afficher « pas de données »."""
    return [dl.TileLayer(url=u, attribution=a, maxZoom=21, maxNativeZoom=z) for u, a, z in FONDS[fond]]


# Charte Bouygues Energies & Services (palette relevée sur bouygues-es.fr)
MARINE, BLEU, CANARD, BLEU_CLAIR = "#1C2959", "#005B90", "#0093BB", "#63BEDD"
VERT, ROUGE, ORANGE = "#00B894", "#E75113", "#F7A600"   # conforme / non conforme / un seul sens mesuré
AMBRE = CANARD                                             # couleur d'accent des graphiques
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

TO93 = Transformer.from_crs(4326, 2154, always_xy=True)
TO84 = Transformer.from_crs(2154, 4326, always_xy=True)

# Thème Plotly commun : fonds transparents, police du dashboard
pio.templates["clair"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["clair"].layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Roboto, Arial, sans-serif", color="#333333", size=12),
    xaxis=dict(gridcolor="#E6E9EF", zerolinecolor="#DDE1E8", linecolor="#BFBFBF"),
    yaxis=dict(gridcolor="#E6E9EF", zerolinecolor="#DDE1E8", linecolor="#BFBFBF"),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=CANARD, font_color="#333333"),
)
pio.templates.default = "clair"
BARRE_COULEUR = dict(thickness=12, x=0.99, xanchor="right", len=0.55, bgcolor="rgba(13,17,23,0.75)",
                     tickfont=dict(color="#e6edf3"), title_font=dict(color="#e6edf3"))

app = dash.Dash(__name__, title="Éclairage public", suppress_callback_exceptions=True,
                compress=True,                    # réponses des callbacks compressées (flask-compress)
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP,
                                      "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap"])


@app.server.route("/rapport/<path:chemin>")
def servir_rapport(chemin):
    """Sert rapport.html / carte.html produits par getys_rapport.py."""
    racine = RAPPORTS_DIR.resolve()
    if not (racine / chemin).is_file():
        abort(404)
    return send_from_directory(racine, chemin)


COUCHES = ("troncons", "hexagones", "points", "lampadaires")


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
    elif couche == "points" and data["mesures"] is not None:
        m = data["mesures"].sort_values("lux")               # les plus chauds dessinés en dernier (dessus)
        h = m["heure_locale"] if "heure_locale" in m else pd.Series(np.nan, index=m.index)
        sens = m["sens"] if "sens" in m else pd.Series(1, index=m.index)
        for lon, lat, e, sn, hh in zip(m["lon"], m["lat"], m["lux"], sens, h):
            f.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
                      "properties": {"E": _r(e), "s": int(sn),
                                     "h": f"{int(hh)}h{int((hh % 1) * 60):02d}" if pd.notna(hh) else ""}})
    elif couche == "lampadaires" and data["luminaires"] is not None:
        lum = data["luminaires"]
        for r in lum.to_dict("records"):
            f.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": {"t": r.get("type", "inconnu"), "lib": r.get("type_libelle", "Luminaire détecté"),
                                     "w": _r(r.get("puissance_w"), 0), "wmin": _r(r.get("puissance_min_w"), 0),
                                     "wmax": _r(r.get("puissance_max_w"), 0), "cote": r.get("cote", "–"),
                                     "pic": _r(r.get("E_pic"), 0), "np": int(r.get("n_passages") or 0)}})
    return {"type": "FeatureCollection", "features": f}


@lru_cache(maxsize=8)
def _couche_gz(dossier, mtime, brut, couche):
    corps = json.dumps(_geojson(charger(dossier, brut=brut), couche), separators=(",", ":")).encode()
    return gzip.compress(corps, 6), f'"{couche}-{int(mtime)}-{int(brut)}"'


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
    sortie = {"resume": {}, "hex": None, "mesures": None, "supports": None, "luminaires": None,
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
    if (d / "supports.csv").exists():
        s = lire_csv(d / "supports.csv")
        if "suspect" in s:
            s["suspect"] = s["suspect"].map(en_bool)
        sortie["supports"] = s
    lum_f = d / "lampadaires_estimes.geojson"                 # analyse actuelle
    if not lum_f.exists():
        lum_f = d / "luminaires_detectes.geojson"            # anciens rapports
    if lum_f.exists():
        gj = json.loads(lum_f.read_text(encoding="utf-8"))
        lum = pd.DataFrame([{**f["properties"], "lon": f["geometry"]["coordinates"][0],
                             "lat": f["geometry"]["coordinates"][1]}
                            for f in gj["features"] if f.get("geometry")])
        sortie["luminaires"] = lum if not lum.empty else None
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


def avec_seuils(tr, sdal):
    """Ajoute le seuil SDAL et la conformité à chaque tronçon."""
    tr = tr.copy()
    tr["seuil"] = tr["classe"].map(lambda c: sdal.get(c, T.CLASSES[c][2]))
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


def zoom_pour(lats, lons, largeur_px=820, hauteur_px=CARTE_H):
    """Zoom MapLibre qui fait tenir l'emprise (avec une marge) dans la carte."""
    lat0 = float(np.mean(lats))
    dx = (max(lons) - min(lons)) * 111320 * math.cos(math.radians(lat0))
    dy = (max(lats) - min(lats)) * 110540
    mpp = max(dx / largeur_px, dy / hauteur_px, 0.15) * 1.35
    return float(np.clip(math.log2(40075016.686 * math.cos(math.radians(lat0)) / (mpp * 512)), 12.5, 17.5))


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
    if mode == "troncons" and couleur == "conformite":
        blocs.append(html.Div([html.Div([html.Span(className="pastille", style={"background": c}), t])
                               for t, c in (("Conforme", VERT), ("Non conforme", ROUGE),
                                            ("Non conforme · 1 seul sens", ORANGE))]))
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
    lum = data["luminaires"] if data else None
    if "lum" in (couches or []) and lum is not None and "type" in lum:
        blocs.append(html.Div([html.Div("Lampadaires estimés", className="leg-titre mt-1")] +
                              [html.Div([html.Span(className="pastille rond", style={"background": COUL_TYPES.get(t, "#fff")}),
                                         f"{t.replace('_', ' ')} ({n})"]) for t, n in lum["type"].value_counts().items()]))
    return blocs


VIDE = {"type": "FeatureCollection", "features": []}


# ------------------------------------------------------------------ PANNEAUX
def deficits_par_rue(tr, n=12):
    """Rues classées par déficit cumulé : Σ longueur × (1 − E/seuil) sur leurs tronçons non conformes.
    Chaque ligne renvoie sur le tronçon le plus sombre de la rue."""
    if tr is None or tr.empty:
        return html.P("Pas de tronçons.", className="text-muted")
    t = tr.assign(manque=(1 - tr["ratio"]).clip(lower=0), lg_nc=tr["longueur_m"].where(~tr["conforme"], 0),
                  e_lg=tr["E_moy"] * tr["longueur_m"], s_lg=tr["seuil"] * tr["longueur_m"])
    t["score"] = t["manque"] * t["longueur_m"]
    g = t.groupby("nom").agg(score=("score", "sum"), lg=("longueur_m", "sum"), lg_nc=("lg_nc", "sum"),
                             e_lg=("e_lg", "sum"), s_lg=("s_lg", "sum"), n=("id", "size"),
                             n_nc=("conforme", lambda c: int((~c).sum())),
                             incomplet=("a_completer", "any"), classe=("classe", lambda c: c.mode().iat[0]))
    g["E"] = g["e_lg"] / g["lg"]
    g["S"] = g["s_lg"] / g["lg"]
    g = g[g["lg_nc"] > 0].sort_values("score", ascending=False).head(n)
    if g.empty:
        return dbc.Alert([html.I(className="bi bi-check-circle me-2"), "Toutes les rues respectent leur seuil SDAL."],
                         color="success", className="mb-0")
    lignes = []
    for rang, (nom, r) in enumerate(g.iterrows(), 1):
        pct = float(min(100, 100 * r["E"] / r["S"])) if r["S"] else 0
        coul = "danger" if pct < 50 else "warning" if pct < 100 else "success"
        # id ASCII : avec des accents, Dash ne retrouve pas la valeur du bouton cliqué (ctx.triggered)
        lignes.append(html.Button(id={"type": "deficit", "index": quote(str(nom))}, n_clicks=0,
                                  className="ligne-deficit", children=[
            html.Div(f"{rang}", className="deficit-rang"),
            html.Div(className="deficit-corps", children=[
                html.Div([html.Span(nom, className="deficit-nom"),
                          html.Span(T.CLASSES[r["classe"]][0], className="badge-type"),
                          html.Span([html.I(className="bi bi-arrow-left-right me-1"), "1 sens"],
                                    className="badge-alerte") if r["incomplet"] else None],
                         className="d-flex align-items-center gap-2 flex-wrap"),
                html.Div(f"{fmt(r['lg_nc'], ' m', 0)} sous le seuil sur {fmt(r['lg'], ' m', 0)} · "
                         f"{r['n_nc']} / {r['n']} tronçon(s)", className="deficit-detail"),
                dbc.Progress(value=pct, color=coul, className="deficit-barre"),
            ]),
            html.Div([html.Div(fmt(r["E"], " lx"), className=f"deficit-valeur text-{coul}"),
                      html.Div(f"seuil {fmt(r['S'], ' lx')}", className="deficit-seuil")], className="text-end"),
        ]))
    return html.Div([
        html.Div("Classement par déficit cumulé (longueur × écart au seuil). Cliquez une rue : la carte "
                 "la cadre en entier et surligne tous ses tronçons.", className="small text-muted mb-2"),
        html.Div(lignes, className="liste-deficits"),
    ])


def distribution(tr):
    """E moyen des tronçons par type de voie, avec le seuil SDAL de chaque type."""
    if tr is None or tr.empty:
        return carte_vide("Pas de tronçons", 360)
    fig = go.Figure()
    for c in [c for c in T.CLASSES if c in set(tr["classe"])]:
        sous = tr[tr["classe"] == c]
        lib = f"{T.CLASSES[c][0]} ({len(sous)})"
        fig.add_trace(go.Box(x=sous["E_moy"], name=lib, boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(size=4, color=AMBRE, opacity=0.6),
                             line_color=MARINE, fillcolor="rgba(28,41,89,0.06)", customdata=sous["nom"],
                             hovertemplate="%{customdata}<br>%{x:.1f} lx<extra></extra>", orientation="h"))
        fig.add_trace(go.Scatter(x=[sous["seuil"].iat[0]] * 2, y=[lib, lib], mode="markers", showlegend=False,
                                 marker=dict(symbol="line-ns", size=30, line=dict(width=3, color=ROUGE)),
                                 hovertemplate=f"seuil SDAL : {sous['seuil'].iat[0]} lx<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                      xaxis_title="Éclairement moyen du tronçon (lx) — trait ambre : seuil SDAL")
    return fig


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
                      xaxis=dict(tickvals=pas, ticktext=[f"{h}h" for h in pas], title="heure locale"))
    return fig


STYLE_TABLE = dict(
    style_table={"overflowX": "auto", "minWidth": "100%"},
    style_header={"backgroundColor": MARINE, "color": "#FFFFFF", "fontWeight": "700",
                  "border": "none", "borderBottom": "2px solid #E75113"},
    style_filter={"backgroundColor": "#F7F8FA", "color": "#333333"},
    style_cell={"backgroundColor": "transparent", "color": "#333333", "border": "none",
                "borderBottom": "1px solid #E6E9EF", "textAlign": "left", "padding": "7px 10px",
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
        html.P("E moy. = moyenne des deux sens de circulation (chaque sens = moyenne de ses passages). "
               "⚠️ : un seul sens mesuré sur une voie à double sens, à repasser. "
               "Cliquez une ligne : la carte va sur le tronçon.", className="small text-muted"),
        dash_table.DataTable(
            id="table-troncons", data=r.to_dict("records"), columns=[{"name": n, "id": c} for c, n in cols],
            sort_action="native", filter_action="native", page_size=12, **STYLE_TABLE,
            style_data_conditional=[
                {"if": {"filter_query": '{conforme} = false'}, "backgroundColor": "rgba(231,81,19,0.08)"},
                {"if": {"state": "active"}, "backgroundColor": "rgba(0,147,187,0.12)", "border": "1px solid #0093BB"}],
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
        html.P("Moyennes fragiles : à repasser ou à vérifier. Cliquez une ligne : la carte va sur le tronçon.",
               className="small text-muted"),
        dash_table.DataTable(
            id="table-qualite", data=fiab, page_size=8, sort_action="native", filter_action="native",
            columns=[{"name": "Voie", "id": "nom"}, {"name": "Problème", "id": "probleme"},
                     {"name": "Détail", "id": "detail"}, {"name": "E moy. (lx)", "id": "E_moy"},
                     {"name": "Long. (m)", "id": "longueur_m"}], **STYLE_TABLE,
            style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "rgba(0,147,187,0.12)",
                                     "border": "1px solid #0093BB"}]),
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
    return html.Div([html.Span(titre, className="outil-titre"), radio], className="outil")


upload_style = {"border": "1.5px dashed #3d444d", "borderRadius": "10px", "padding": "22px",
                "textAlign": "center", "cursor": "pointer"}

champs_sdal = [
    dbc.Row(className="g-2 align-items-center mb-1", children=[
        dbc.Col(html.Span(lib, className="small"), width=7),
        dbc.Col(dbc.InputGroup([dbc.Input(id=f"sdal-{c}", type="number", value=v, min=0, step=0.5, debounce=True,
                                          size="sm", className="text-end"),
                                dbc.InputGroupText("lx", className="py-0 small")], size="sm"), width=5),
    ]) for c, (lib, _, v) in T.CLASSES.items()]

app.layout = html.Div(className="page", children=[
    dcc.Store(id="selection"), html.Div(id="defilement", hidden=True),
    dcc.Store(id="urls-prechargement"), html.Div(id="prechargement", hidden=True),
    # --- barre du haut
    html.Header(className="entete", children=dbc.Container(fluid=True, children=dbc.Row(
        align="center", className="g-2", children=[
            dbc.Col(xs=12, lg=True, children=html.Div([
                # logo officiel (repris du dossier du hackathon, Getys_Hackathon.pdf)
                html.A(html.Img(src=app.get_asset_url("logo-bouygues-es.png"), alt="Bouygues Energies & Services"),
                       href=LIEN_HACKATHON, target="_blank", rel="noopener", className="logo-entreprise",
                       title="Bouygues Energies & Services"),
                html.Div([html.H1("Clair-Obscur", className="titre mb-0"),
                          html.Div("Tableau de bord de l'éclairage public", className="sous-titre")]),
            ], className="d-flex align-items-center gap-3")),
            dbc.Col(xs=12, md=6, lg=4, children=dcc.Dropdown(
                id="choix-rapport", clearable=False, placeholder="Choisir une tournée…",
                className="choix-rapport")),
            dbc.Col(xs="auto", children=html.A([html.I(className="bi bi-file-earmark-text me-1"),
                                                 "Rapport complet"], id="lien-rapport", target="_blank",
                                                className="btn btn-outline-light btn-sm disabled")),
            dbc.Col(xs="auto", children=html.A([html.I(className="bi bi-globe-europe-africa me-1"),
                                                 "Carte satellite"], id="lien-carte", target="_blank",
                                                className="btn btn-outline-light btn-sm disabled")),
            dbc.Col(xs="auto", children=dbc.Button([html.I(className="bi bi-cloud-arrow-up me-1"),
                                                     "Nouvelle tournée"], id="btn-nouvelle",
                                                    color="warning", size="sm", className="fw-semibold")),
        ]))),

    dbc.Container(fluid=True, className="contenu", children=[
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
                        {"label": [html.I(className="bi bi-thermometer-half me-1"), "Points chauds / froids"],
                         "value": "points"}], "Vue"),
                    segmente("couleur-carte", "eclairement", [
                        {"label": "Éclairement", "value": "eclairement"},
                        {"label": "Conformité SDAL", "value": "conformite"}], "Couleur (tronçons)"),
                    segmente("lissage", "apres", [
                        {"label": [html.I(className="bi bi-soundwave me-1"), "Avant"], "value": "avant"},
                        {"label": [html.I(className="bi bi-activity me-1"), "Après lissage"], "value": "apres"}],
                        "Bruit"),
                    segmente("fond", "clair", [
                        {"label": [html.I(className="bi bi-moon-stars me-1"), "Sombre"], "value": "sombre"},
                        {"label": [html.I(className="bi bi-sun me-1"), "Clair"], "value": "clair"},
                        {"label": [html.I(className="bi bi-globe-europe-africa me-1"), "Satellite"],
                         "value": "satellite"}], "Fond"),
                    dbc.Checklist(id="couches", value=[], switch=True, className="small outil-couches",
                                  options=[{"label": "Lampadaires estimés", "value": "lum"},
                                           {"label": "Supports suspects", "value": "sus"}]),
                ])),
                html.Div(id="bandeau-lissage"),
                dbc.CardBody(html.Div(className="carte-leaflet", children=[
                    dl.Map(id="carte", center=[48.785, 2.54], zoom=14.5, zoomSnap=0.25, zoomDelta=0.5,
                           preferCanvas=True, maxZoom=21, style={"height": f"{CARTE_H}px", "width": "100%"},
                           children=[
                               dl.LayerGroup(id="fonds", children=tuiles("clair")),
                               dl.GeoJSON(id="geo-troncons", data=VIDE, style=fn("styleTroncon"),
                                          onEachFeature=fn("infoTroncon"), hideout={}),
                               dl.GeoJSON(id="geo-hex", data=VIDE, style=fn("styleHex"), onEachFeature=fn("infoHex")),
                               dl.GeoJSON(id="geo-points", data=VIDE, pointToLayer=fn("pointChaud")),
                               dl.GeoJSON(id="geo-lampes", data=VIDE, pointToLayer=fn("lampe"),
                                          onEachFeature=fn("infoLampe")),
                               dl.LayerGroup(id="groupe-selection"),
                               dl.ScaleControl(position="bottomright", imperial=False),
                           ]),
                    html.Div(id="legende", className="legende-carte"),
                ]), className="p-0"),
                dbc.CardFooter([html.I(className="bi bi-cursor me-2"),
                                "Cliquez sur la carte, une rue du classement ou une ligne d'un tableau : "
                                "la carte va sur le tronçon et Street View s'affiche à droite."],
                               className="small text-muted"),
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
                    html.Div("Éclairement moyen minimal exigé (hypothèses en attendant le SDAL de la commune). "
                             "Tout le dashboard se met à jour.", className="small text-muted mt-2"),
                ]),
            ]),
        ]),

        dbc.Row(className="g-3 mt-0", children=[
            dbc.Col(xl=5, children=carte_bloc("Rues les plus en déficit", "sort-down",
                                              html.Div(id="deficits"))),
            dbc.Col(xl=7, children=dbc.Card(className="bloc h-100", children=[
                dbc.CardHeader(dbc.Tabs(id="onglets", active_tab="troncons", className="card-header-tabs", children=[
                    dbc.Tab(label="Tronçons", tab_id="troncons"),
                    dbc.Tab(label="Par type de voie", tab_id="distrib"),
                    dbc.Tab(label="Lampadaires", tab_id="lampes"),
                    dbc.Tab(label="Chronologie", tab_id="chrono"),
                    dbc.Tab(label="Qualité des données", tab_id="qualite"),
                ])),
                dbc.CardBody([
                    html.Div(id="tableau"),
                    html.Div(dcc.Graph(id="distrib", config={"displayModeBar": False}), id="p-distrib"),
                    html.Div(dcc.Graph(id="lampes", config={"displayModeBar": False}), id="p-lampes"),
                    html.Div(dcc.Graph(id="chrono", config={"displayModeBar": False}), id="p-chrono"),
                    html.Div(id="qualite"),
                ]),
            ])),
        ]),
        html.Footer([html.A("Hackathon 2026 Bouygues Energies & Services", href=LIEN_HACKATHON, target="_blank",
                            rel="noopener"), " · équipe Clair-Obscur · mesures GETYS · lampadaires estimés et "
                     "palette de carte.html (getys_rapport.py) · filaire © contributeurs OpenStreetMap"],
                    className="pied"),
    ]),

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
            html.P("Seule entrée : le fichier .tdms. L'analyse (≈ 10 s) estime les lampadaires, puis le "
                   "dashboard calcule la moyenne au tronçon (filaire OpenStreetMap) et les hexagones.",
                   className="small text-muted mt-3 mb-0"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Annuler", id="btn-annuler", color="secondary", outline=True),
            dbc.Button([html.I(className="bi bi-play-fill me-1"), "Lancer l'analyse"], id="btn-lancer",
                       color="warning", disabled=True, className="fw-semibold"),
        ]),
    ]),
])

SDAL_INPUTS = [Input(f"sdal-{c}", "value") for c in T.CLASSES]
SDAL_STATES = [State(f"sdal-{c}", "value") for c in T.CLASSES]


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
              State("choix-rapport", "value"), prevent_initial_call="initial_duplicate")
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


@app.callback(Output("lien-rapport", "href"), Output("lien-rapport", "className"),
              Output("lien-carte", "href"), Output("lien-carte", "className"), Output("urls-prechargement", "data"),
              Input("choix-rapport", "value"))
def liens_rapport(dossier):
    """Liens vers rapport.html et carte.html produits par getys_rapport.py + URL des hexagones à précharger."""
    classe = "btn btn-outline-light btn-sm"
    sortie = []
    for f in ("rapport.html", "carte.html"):
        if dossier and (Path(dossier) / f).exists():
            sortie += [f"/rapport/{Path(dossier).name}/{f}", classe]
        else:
            sortie += [None, classe + " disabled"]
    if not dossier:
        return sortie + [None]
    data = charger(dossier)
    return sortie + [[url_couche(data, c) for c in ("hexagones", "points")]]


# données des vues Hexagones / Points téléchargées en arrière-plan : elles s'ouvrent sans attente
app.clientside_callback(
    """function(urls) {
        (urls || []).forEach((u, i) => setTimeout(() => fetch(u).catch(() => null), 1500 + 1000 * i));
        return '';
    }""",
    Output("prechargement", "children"), Input("urls-prechargement", "data"))


ONGLETS = {"troncons": "tableau", "distrib": "p-distrib", "lampes": "p-lampes", "chrono": "p-chrono",
           "qualite": "qualite"}


@app.callback(*[Output(p, "style") for p in ONGLETS.values()], Input("onglets", "active_tab"))
def changer_onglet(onglet):
    return [{"display": "block" if o == onglet else "none"} for o in ONGLETS]


# ------------------------------------------------------------------ CALLBACKS : AFFICHAGE
@app.callback(Output("contexte", "children"), Output("kpis", "children"), Output("deficits", "children"),
              Output("distrib", "figure"), Output("lampes", "figure"), Output("chrono", "figure"),
              Output("tableau", "children"), Output("qualite", "children"),
              Input("choix-rapport", "value"), *SDAL_INPUTS)
def afficher(dossier, *valeurs):
    if not dossier:
        vide = carte_vide("Aucune tournée analysée — cliquez sur « Nouvelle tournée »", 360)
        return None, [], None, vide, vide, vide, None, None
    sdal = sdal_de(valeurs)
    data = charger(dossier)
    meta = data["resume"].get("meta", {})
    k = data["resume"].get("kpi", {})
    tr = avec_seuils(data["troncons"], sdal) if data["troncons"] is not None else None

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

    return ([contexte, alerte], kpis, deficits_par_rue(tr), distribution(tr), types_lampes(data),
            chronologie(data), tableau_troncons(tr), qualite(data, tr))


# ------------------------------------------------------------------ CALLBACKS : CARTE (Leaflet)
# Chaque calque charge ses données par URL (cache navigateur). Le mode affiche un calque et vide les autres ;
# la couleur et les seuils SDAL ne changent que le « hideout » des tronçons (recoloriage dans le navigateur).
# Leaflet garde sa vue : rien ne bouge quand on change de filtre.
@app.callback(Output("geo-troncons", "url"), Output("geo-troncons", "data"),
              Output("geo-hex", "url"), Output("geo-hex", "data"),
              Output("geo-points", "url"), Output("geo-points", "data"),
              Output("bandeau-lissage", "children"),
              Input("choix-rapport", "value"), Input("mode-carte", "value"), Input("lissage", "value"))
def afficher_calques(dossier, mode, lissage):
    if not dossier:
        return None, VIDE, None, VIDE, None, VIDE, None
    brut = lissage == "avant"
    data = charger(dossier, brut=brut)
    sortie = []
    for m, couche in (("troncons", "troncons"), ("hex", "hexagones"), ("points", "points")):
        sortie += [url_couche(data, couche), None] if mode == m else [None, VIDE]
    bandeau = []
    if mode == "hex":
        bandeau.append(html.Div([html.I(className="bi bi-zoom-in me-2"),
                                 "Hexagones de 2 m : bande continue le long du trajet (3 luxmètres à ±0,49 m). "
                                 "Zoomez sur un quartier pour les distinguer."], className="bandeau-lissage"))
    if mode == "troncons" and data["troncons"] is None:
        bandeau.append(html.Div("Filaire de voirie indisponible (réseau ?) : " + (data["erreur_troncons"] or ""),
                                className="bandeau-lissage"))
    if brut:
        texte = ("Avant lissage : signal brut (médiane des 3 capteurs, sans filtre de Hampel ni suppression des pics "
                 "aberrants). Comparez avec « Après lissage ».")
        if not data["brut"]:
            texte = "Ce rapport ne contient pas le signal brut (colonne E_brut) : relancez l'analyse pour l'obtenir."
        bandeau.append(html.Div([html.I(className="bi bi-soundwave me-2"), texte], className="bandeau-lissage"))
    return (*sortie, bandeau or None)


@app.callback(Output("geo-troncons", "hideout"), Input("couleur-carte", "value"), Input("mode-carte", "value"),
              *SDAL_INPUTS)
def colorer_troncons(couleur, mode, *valeurs):
    """Mode de couleur et seuils SDAL : appliqués par la fonction de style dans le navigateur."""
    return {"couleur": couleur if mode == "troncons" else "eclairement", "sdal": sdal_de(valeurs)}


@app.callback(Output("geo-lampes", "url"), Output("geo-lampes", "data"),
              Input("couches", "value"), Input("choix-rapport", "value"))
def afficher_lampes(couches, dossier):
    if dossier and "lum" in (couches or []):
        return url_couche(charger(dossier), "lampadaires"), None
    return None, VIDE


@app.callback(Output("legende", "children"), Input("mode-carte", "value"), Input("couleur-carte", "value"),
              Input("couches", "value"), Input("choix-rapport", "value"))
def afficher_legende(mode, couleur, couches, dossier):
    return legende(mode, couleur if mode == "troncons" else "eclairement", couches,
                   charger(dossier) if dossier else None)


@app.callback(Output("fonds", "children"), Output("carte", "className"), Input("fond", "value"))
def changer_fond(fond):
    return tuiles(fond), f"fond-{fond}"


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


def _selection_rue(data, nom):
    """Toute la rue : tronçons surlignés, carte cadrée sur la rue, Street View au point le plus sombre."""
    tr = data["troncons"]
    rue = tr[tr["nom"] == nom]
    if rue.empty:
        return dash.no_update
    tr_lat, tr_lon = [], []
    for tid in rue["id"]:
        glon, glat = data["geoms"][tid]
        tr_lat += list(glat) + [None]; tr_lon += list(glon) + [None]
    lats = [v for v in tr_lat if v is not None]
    lons = [v for v in tr_lon if v is not None]
    lat, lon, cap, e_min = point_le_plus_sombre(data, *rue["id"])
    lg = float(rue["longueur_m"].sum())
    e = float((rue["E_moy"] * rue["longueur_m"]).sum() / lg)
    return {"lat": lat, "lon": lon, "cap": cap, "titre": nom, "centrer": True, "tid": f"rue:{nom}",
            "detail": f"rue entière : {len(rue)} tronçon(s) · {fmt(lg, ' m', 0)} · E moyen {fmt(e, ' lx')} · "
                      f"Street View au point le plus sombre ({fmt(e_min, ' lx')})",
            "tr_lat": tr_lat, "tr_lon": tr_lon,
            "c_lat": (min(lats) + max(lats)) / 2, "c_lon": (min(lons) + max(lons)) / 2,
            "zoom": zoom_pour(lats, lons) + 1}


@app.callback(Output("selection", "data"),
              Input("geo-troncons", "clickData"), Input("geo-hex", "clickData"), Input("geo-points", "clickData"),
              Input("geo-lampes", "clickData"), Input({"type": "deficit", "index": ALL}, "n_clicks"),
              Input("table-troncons", "active_cell"), Input("table-qualite", "active_cell"),
              State("choix-rapport", "value"), prevent_initial_call=True)
def choisir_point(clic_tr, clic_hex, clic_pt, clic_lampe, clics_deficit, cellule, cellule_q, dossier):
    if not dossier:
        return dash.no_update
    data = charger(dossier)
    src = ctx.triggered_id
    clics = {"geo-troncons": clic_tr, "geo-hex": clic_hex, "geo-points": clic_pt, "geo-lampes": clic_lampe}
    clic = clics.get(src) if isinstance(src, str) else None       # src est un dict pour la liste des déficits
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
        elif src == "geo-lampes":
            sel.update(titre=f"💡 Lampadaire estimé — {p.get('lib')}",
                       detail=f"{fmt(p.get('w'), ' W', 0)} estimés · côté {p.get('cote')} · pic {fmt(p.get('pic'), ' lx', 0)}")
        elif src == "geo-points":
            sel.update(titre="Point de mesure", detail=f"{fmt(p.get('E'), ' lx')} · sens {p.get('s')} · {p.get('h')}")
        elif src == "geo-hex":
            sel.update(titre="Hexagone de 2 m", detail=f"{fmt(p.get('E'), ' lx')} (sens 1 : {fmt(p.get('s1'), ' lx')}, "
                                                       f"sens 2 : {fmt(p.get('s2'), ' lx')}) · {p.get('np')} passage(s)")
        return sel
    tid, extra = None, None
    if isinstance(src, dict) and src.get("type") == "deficit":
        if not ctx.triggered or not ctx.triggered[0]["value"]:      # liste redessinée, pas un clic
            return dash.no_update
        if data["troncons"] is None:
            return dash.no_update
        return _selection_rue(data, unquote(src["index"]))
    elif src == "table-troncons" and cellule:
        tid = cellule.get("row_id")
    elif src == "table-qualite" and cellule_q and cellule_q.get("row_id"):
        tid = cellule_q["row_id"].split("|")[0]
        extra = "à fiabiliser"
    if tid is None or data["geoms"] is None or tid not in data["geoms"]:
        return dash.no_update
    return _selection_troncon(data, tid, extra)


@app.callback(Output("sv-cadre", "children"), Output("sv-legende", "children"),
              Output("sv-externe", "href"), Output("sv-externe", "className"),
              Input("selection", "data"), prevent_initial_call=True)
def afficher_streetview(sel):
    if not sel:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    cadre = html.Iframe(src=embed_streetview(sel["lat"], sel["lon"], sel.get("cap", 0)), className="sv-iframe",
                        allow="fullscreen")
    legende = [html.Div(sel.get("titre"), className="fw-semibold"),
               html.Div(sel.get("detail") or "", className="small text-muted"),
               html.Div(f"{sel['lat']:.5f}, {sel['lon']:.5f} · Street View montre la rue de jour : utile pour "
                        "vérifier la présence de lampadaires, pas s'ils sont allumés.", className="small text-muted")]
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
                     "zoom": sel.get("zoom", 17.5), "transition": "flyTo"}


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


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8050")),
            debug=os.environ.get("DEBUG", "1") == "1")
