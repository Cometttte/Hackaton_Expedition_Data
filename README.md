# Outil GETYS : rapport automatique de fin de tournée

**Seule entrée : le fichier `.tdms` de la tournée.** Aucun inventaire n'est nécessaire.

## Installation
    pip install npTDMS pandas numpy scipy pyproj

## Utilisation
Une tournée :

    python getys_rapport.py Ormesson-sur-Marne.tdms -o rapports

Surveillance d'un dossier (un rapport est produit dès qu'un nouveau .tdms est complet) :

    python getys_rapport.py --surveiller D:/acquisitions -o rapports

Tous les seuils sont réglables en ligne de commande (`--dist-laterale-m 4.5`, `--troncon-m 50`, `--seuil-emoy-lx 7.5`, etc. ; voir `--help`).

## Sorties (rapports/<nom>_<date>_<fichier>/)
Si un rapport existe déjà pour la même tournée, il est déplacé dans `rapports/archives/<nom>_genere_<date-heure>/`. Rien n'est jamais écrasé.

- `rapport.html` : rapport autonome. Il contient une carte zoomable, les lampadaires estimés, le type de lampe, la puissance et la consommation, la fiabilité des estimations, la qualité des mesures et l'abaissement nocturne.
- `carte.html` : carte sur fond satellite. Calques : points chauds / froids, 3 capteurs filtrés par sens de passage, lampadaires estimés (par type ou par côté), tronçons à lampadaires des deux côtés. Le fond de carte demande une connexion internet.
- `lampadaires_estimes.csv` / `.geojson` : un lampadaire par ligne, avec sa position, son côté, le type de lampe, la Tcp, la puissance et sa fourchette, et la consommation.
- `mesures_pas2m.csv` : un point tous les 2 m roulés (lon/lat, Lambert 93, médiane et 3 capteurs, signal brut avant filtrage `E_brut`, vitesse, cap, sens, heure).
- `resume.json` : tous les indicateurs.

## Chaîne de traitement (≈ 6 s pour 3 h 20 de relevé)
1. Lecture du TDMS. Les capteurs sont sur une remorque, à 1,5 m du sol. Le GNSS vient du port C (5 Hz, avec cap IMU), avec repli sur le port A.
2. Correction de l'horloge PC / GPS, puis calage automatique des capteurs sur la trace (latence et bras de levier, estimés sur les allers-retours).
3. Filtrage du bruit : négatifs, filtre de Hampel, médiane des 3 luxmètres, pics aberrants et passages incohérents.
4. Ré-échantillonnage à un point tous les 2 m, hors arrêts. Sens 1 / sens 2 calculé sur l'axe local de la voie.
5. Lampadaires : pics de lumière regroupés sur tous les passages, puis position le long de la voie, côté gauche / droite par l'asymétrie des luxmètres, et implantation un côté / deux côtés par tronçon de 100 m.
6. Type de lampe d'après le chromamètre (sodium, blanc 2 700 K, LED 3 000 K, blanc neutre, blanc froid), puis puissance et consommation par type, avec une fourchette.
7. Abaissement nocturne : pic d'un même lampadaire avant et après 22h.

Le détail de chaque déduction, et la précision constatée sur Ormesson, sont dans EXPLICATION_DU_CODE.md.

## Dashboard (dossier `dashboard/`)
Tableau de bord web des tournées : moyenne au tronçon, hexagones, densité, conformité SDAL, lampadaires estimés et Street View intégré.

    docker compose up --build -d        # puis http://localhost:8050

Sans Docker :

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/flask --app dashboard/serveur run --port 5001    # serveur d'analyse (bouton « Nouvelle tournée »)
    .venv/bin/python dashboard/app.py                          # dashboard sur http://localhost:8050

Le dashboard lit `rapports/` et calcule à partir de `mesures_pas2m.csv` :
- **Moyenne au tronçon** (`dashboard/troncons.py`) : filaire de voirie OpenStreetMap (cache dans `dashboard/voirie/`), coupé à chaque carrefour (parts de ~150 m au-delà de 250 m). Chaque mesure est rattachée au tronçon le plus proche (< 15 m) d'orientation compatible avec le cap. Moyenne **par sens de passage** (chaque passage pèse autant), puis **moyenne des deux sens**. Un tronçon à double sens mesuré dans un seul sens est signalé « à repasser ».
- **Hexagones de 10 m** (`dashboard/hexagones.py`) : même règle, une moyenne par sens puis moyenne des deux sens.
- **Conformité SDAL** : un seuil d'éclairement moyen par type de voie (principale, distribution, résidentielle, desserte), modifiable dans l'interface. Valeurs par défaut = hypothèses en attendant le SDAL de la commune.
- **Points chauds / froids** : même méthode et même palette (0 à 50 lx) que `carte.html` : valeur moyenne, rayon de 9 m quel que soit le zoom, filtre sens 1 / sens 2.
- **Avant / après lissage du bruit** : toutes les vues peuvent être recalculées sur le signal brut (colonne `E_brut` de `mesures_pas2m.csv` : médiane des 3 capteurs, sans filtre de Hampel ni suppression des pics aberrants).
- **Fond clair / sombre**, couches (lampadaires estimés) et sélection mis à jour sans recharger la carte : le zoom est conservé.
- **Rues les plus en déficit** : classement par déficit cumulé (longueur × écart au seuil SDAL) ; un clic centre la carte sur le tronçon le plus sombre.
- **Street View** : clic sur la carte, le classement, le tableau des tronçons ou les tronçons à fiabiliser (onglet Qualité) ; la carte se centre sur le tronçon surligné et la vue s'oriente dans le sens de la tournée.
