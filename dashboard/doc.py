"""Onglet « Doc » : techniques et algorithmes utilisés, stack technique (texte Markdown)."""
from pathlib import Path

ALGORITHMES = r"""
### 1. Lecture et synchronisation (`getys_analyse.py`)
- **Lecture du TDMS** (npTDMS) : 3 luxmètres à 50 Hz, GNSS du port C (5 Hz, cap de la centrale inertielle) avec repli sur le port A (1 Hz), chromamètre (Ev, Tcp, Δuv).
- **Correction d'horloge** : décalage PC → UTC GPS estimé par la médiane des écarts entre horodatage de réception et heure des trames GGA.
- **Interpolation des positions** sur les instants des luxmètres ; pas d'interpolation au travers d'un trou GNSS de plus de 2 s.
- **Calage capteurs / GPS** : latence et bras de levier de la remorque estimés automatiquement sur les allers-retours (recherche du décalage qui superpose le mieux les pics des deux sens).

### 2. Filtrage du bruit
- Valeurs négatives ramenées à 0.
- **Filtre de Hampel** (médiane glissante ± k × MAD) sur chaque capteur : supprime les parasites d'un seul capteur.
- **Médiane des 3 luxmètres** : signal de référence robuste à un capteur défaillant.
- **Pics aberrants** retirés puis interpolés : trop hauts, trop brefs (largeur < quelques mètres à la vitesse mesurée) ou variations à l'arrêt.
- Passages incohérents écartés (phares, vitrine) par comparaison entre passages.

### 3. Spatialisation
- **Ré-échantillonnage spatial** : un point tous les 2 m parcourus (moyenne des échantillons du pas), arrêts exclus.
- **Sens 1 / sens 2** : axe local de la voie (moyenne axiale des caps, angles doublés) dans des mailles de 30 m.
- **Hexagones de 2 m** (`dashboard/hexagones.py`) : un point par luxmètre décalé latéralement (±0,49 m), interpolation tous les 0,5 m, puis moyenne par passage → par sens → moyenne des deux sens.

### 4. Moyenne au tronçon et conformité (`dashboard/troncons.py`)
- **Filaire OpenStreetMap** (API Overpass, mis en cache), découpé de croisement à croisement.
- **Map-matching** : chaque mesure est rattachée au tronçon le plus proche (< 15 m) dont l'orientation est compatible avec le cap du véhicule.
- **Moyenne par sens** (chaque passage pèse autant), puis **moyenne des deux sens** : les deux côtés de la rue comptent autant. Un tronçon à double sens mesuré dans un seul sens est signalé « à repasser ».
- **Conformité SDAL** : éclairement moyen du tronçon comparé au seuil de son type de voie, ou au seuil propre de la rue si un fichier SDAL par rue est importé. Carte : bleu sous le seuil, vert conforme jusqu'à 40 lx, rouge au-delà de 40 lx.

### 5. Heatmap (`dashboard/heatmap_flux.py`)
- **Lissage gaussien normalisé** (σ = 4 m) des moyennes des cellules : Σ G·E / Σ G, sans assombrir les bords ; surface coupée à 6 m des mesures (pas d'extrapolation).
- **Image en dégradé** rééchantillonnée en Web Mercator (pixels de 1 m), transparence adoucie au bord.
- **Isobandes vectorielles** (export GeoJSON) : classes d'éclairement, contours tracés sur la grille (4-connexité), trous compris.

### 6. Lampadaires estimés
- **Détection de pics** (`scipy.signal.find_peaks`) sur chaque passage : proéminence ≥ 3 lx et ≥ 30 % de la hauteur, pics espacés d'au moins 8 m.
- **Regroupement des passages** à moins de 6 m ; **côté gauche / droite** par l'asymétrie entre luxmètres gauche et droit, votée sur tous les passages.
- **Type de lampe** par le chromamètre (Tcp, Δuv), puis **puissance** moyenne du type avec sa fourchette.

### 7. Contrôle opérateur (vue Opérateur)
- Lecture brute du TDMS **sans analyse** (≈ 3 s) : à chaque position GPS (5 Hz), la valeur de chacun des 3 capteurs à cet instant, sans moyenne, lissage ni filtre ; position latérale des capteurs lue dans le TDMS. L'horloge du PC est seulement recalée sur l'heure GPS pour situer les mesures.
- Contrôles : coupures GNSS > 2 s, HDOP, capteur figé ou négatif, désaccord d'un capteur (> 50 % des deux autres).
"""

STACK = r"""
| Couche | Technologies |
|---|---|
| Analyse | Python 3.12 · NumPy · pandas · SciPy (`signal`, `ndimage`, `spatial.cKDTree`) · pyproj (Lambert 93 ↔ WGS84 ↔ Web Mercator) · npTDMS |
| Dashboard | Dash 3 · Dash Bootstrap Components · dash-leaflet (Leaflet) · Plotly · dash_table |
| Cartographie | Leaflet + MapLibre GL (fond vectoriel OpenFreeMap) · tuiles Esri (satellite, fonds gris) · filaire OpenStreetMap (Overpass) |
| Heatmap autonome | MapLibre GL JS · PNG RGBA et GeoJSON générés en Python |
| Serveur | Flask (serveur d'analyse des `.tdms`) · flask-compress (gzip) |
| Déploiement | Docker · Docker Compose (services `serveur` et `dashboard`) |
| Données | `.tdms` en entrée ; sorties CSV, GeoJSON, JSON, HTML par tournée (`rapports/`) |
"""


def texte_complet() -> str | None:
    """Explication détaillée du code (EXPLICATION_DU_CODE.md), si elle est présente."""
    f = Path(__file__).resolve().parent.parent / "EXPLICATION_DU_CODE.md"
    return f.read_text(encoding="utf-8") if f.exists() else None
