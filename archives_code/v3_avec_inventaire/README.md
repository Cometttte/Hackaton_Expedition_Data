# Outil GETYS : rapport automatique de fin de tournée

## Installation
    pip install npTDMS pandas openpyxl numpy scipy pyproj

## Utilisation
Une tournée :

    python getys_rapport.py Ormesson-sur-Marne.tdms ORMASSON-exportations.ouvrages.xlsx -o rapports

Surveillance d'un dossier (un rapport est produit dès qu'un nouveau .tdms est complet) :

    python getys_rapport.py --surveiller D:/acquisitions ORMASSON-exportations.ouvrages.xlsx -o rapports

Tous les seuils sont réglables en ligne de commande (`--hex-m 15`, `--seuil-emoy-lx 7.5`, `--seuil-eteint-lx 5`, etc. ; voir `--help`).

## Sorties (rapports/<ville>_<date>_<fichier>/)
Si un rapport existe déjà pour la même tournée, il est déplacé dans `rapports/archives/<nom>_genere_<date-heure>/` : rien n'est jamais écrasé.

- `rapport.html` : rapport autonome (carte zoomable, constats, tableaux)
- `carte.html` : carte sur fond satellite avec calques : points chauds / froids, 3 capteurs filtrés par sens de passage, hexagones, conformité, supports suspects, sources non inventoriées (connexion internet nécessaire pour le fond de carte)
- `mesures_pas2m.csv` : un point tous les 2 m roulés (lon/lat, Lambert 93, lux filtré, 3 capteurs, heure, rue)
- `hexagones.geojson` : cellules de 10 m (E moy., médiane, P10, min, max, U0, nb de passages, conformité)
- `supports.csv` : chaque support avec distance, max au passage, suspect, source mesurée au chromamètre
- `positions_a_corriger.csv` : supports dont la position dans l'Excel est fausse de plus de 5 m, avec la position mesurée (prêt pour le SIG)
- `rues.csv`, `luminaires_detectes.geojson`, `resume.json`

## Chaîne de traitement (≈ 5 s pour 3 h 20 de relevé)
1. Lecture TDMS (capteurs sur remorque à 1,5 m du sol) ; GNSS 5 Hz du port C (avec cap IMU), repli sur le port A 1 Hz.
2. Correction du décalage d'horloge PC / GPS, puis calage automatique des capteurs sur la trace (latence d'horodatage + bras de levier antenne / remorque, estimés sur les allers-retours).
3. Bruit : négatifs ramenés à 0, Hampel 0,5 s par capteur, médiane des 3 luxmètres, puis retrait des pics aberrants (> 150 lx, ou largeur à mi-hauteur < 2 m) et des passages incohérents. Détail dans EXPLICATION_DU_CODE.md.
4. Spatialisation par interpolation, exclusion des arrêts, ré-échantillonnage à pas de distance (2 m).
5. Hexagones et multi-passages : moyenne par passage, puis moyenne des passages.
6. Supports : max au passage, suspects éteints, armoires à risque, rues.
7. Chromamètre : source sodium / blanche au passage, écarts avec l'inventaire.
8. Détection des luminaires (pics) et côté gauche/droite. Les supports de l'Excel sont recalés sur les points lumineux mesurés : décalage d'ensemble, puis affectation un pour un et correction le long de la voie. Les mesures par support sont refaites à la position recalée. Les positions fausses de plus de 5 m sont signalées. Les sources non inventoriées sont aussi listées.
9. Abaissement nocturne : comparaison appariée avant / après 22h sur les mêmes cellules.
