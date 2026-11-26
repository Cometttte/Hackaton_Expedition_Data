# Explication du code : outil de rapport de fin de tournée GETYS

Ce document décrit le fonctionnement de l'outil étape par étape. Pour chaque déduction, il indique ce qu'on en conclut, sur quoi la conclusion repose (physique, données observées) et le paramètre qui la règle.
Les chiffres d'exemple viennent de la tournée d'Ormesson-sur-Marne du 18/03/2025.

**Seule entrée : le fichier `.tdms`.** Depuis la version 4, l'outil n'utilise plus l'export Excel des ouvrages. Les lampadaires, leur côté, leur type de lampe et leur puissance sont estimés à partir des mesures seules (sections 7 à 9). L'inventaire d'Ormesson n'a servi qu'une fois, pour mesurer la précision de ces estimations (section 10).

---

## 0. Organisation des fichiers

| Fichier | Rôle |
|---|---|
| `getys_rapport.py` | Point d'entrée en ligne de commande. Il lance l'analyse, écrit le rapport HTML et peut surveiller un dossier pour traiter chaque nouvelle tournée. |
| `getys_analyse.py` | Toute l'analyse. La fonction `analyser()` enchaîne les étapes décrites ci-dessous et renvoie un dictionnaire de résultats. |
| `carte_template.html` | Gabarit de la carte satellite (`carte.html`). |
| `rapport_template.html` | Gabarit du rapport. Le script y injecte les résultats en JSON. Les textes du rapport sont générés à partir des chiffres, donc ils s'adaptent à chaque tournée. |
| `Params` (dans `getys_analyse.py`) | Tous les seuils au même endroit. Chacun se règle en ligne de commande (`--dist-laterale-m 4.5`, par exemple). |

Durée totale sur la tournée d'exemple (3h20, 72 km) : environ 5 secondes. Les calculs sont vectorisés avec numpy et scipy, sans boucle Python sur les 616 000 échantillons. La seule boucle porte sur les quelque 2 000 pics de lumière détectés.

---

## 1. Lecture du fichier TDMS (`TdmsFile.read`)

Le fichier contient les groupes suivants :

| Groupe | Contenu | Utilisation |
|---|---|---|
| `Luxmètres` | Lux1, Lux2, Lux3 à 50 Hz, avec propriétés `lux_offset` et `lux_position` | Mesure principale |
| `GNSS` | Port A : trames NMEA `$GPGGA` à 1 Hz, avec l'heure PC de réception de chaque trame | Calcul du décalage d'horloge, et position de secours |
| `IMU` | Port C : flux d'octets avec `$GPGGA` à 5 Hz et `$PRDID` (tangage, roulis, cap) | Position principale |
| `Calorimetre` | Ev (lux), Tcp (température de couleur), deltaUV, à environ 1 Hz | Type de source |
| `Voies analogiques` | Mod1/ai3 en volts, autour de 12 V | Non utilisé (probablement la tension de la batterie) |

**Déduction : le port C est un GNSS 5 Hz.** Le groupe s'appelle « IMU », mais une fois décodé en texte il contient des trames GGA cinq fois par seconde, avec le cap de la centrale inertielle dans les trames `$PRDID`. Comme il est cinq fois plus dense que le port A, l'outil le prend en priorité. Il revient au port A si le port C est absent (fonction `load_gnss`).

---

## 2. Synchronisation des horloges (`load_gnss`)

**Constat.** Les luxmètres sont horodatés par l'horloge du PC (`wf_start_time` + n × 0,02 s), et les positions par l'heure UTC du GPS. Ces deux horloges ne sont pas d'accord.

**Méthode.** Pour chaque trame GGA du port A, on dispose de l'heure PC de réception et de l'heure UTC contenue dans la trame. La médiane de « heure PC − heure GPS » donne le décalage. On prend la médiane pour ne pas être sensible aux retards ponctuels de réception.

**Résultat.** Le PC avance de 237,5 s sur le GPS. Sans cette correction, chaque mesure serait placée environ 4 minutes plus loin sur le trajet, soit plusieurs kilomètres.

**Heure locale.** On part de l'UTC du GPS, puis on convertit en Europe/Paris (UTC+1 en mars). La tournée va de 20h17 à 23h38.

---

## 2 bis. Calage des capteurs de la remorque sur la trace GPS (`calage_longitudinal`)

**Problème.** Le GPS donne la position de son antenne, à l'instant de son horodatage. La lumière, elle, est mesurée par les capteurs de la remorque. Deux erreurs peuvent placer chaque mesure trop en avant ou trop en arrière sur la route :
- **un bras de levier fixe** : la distance, le long de la route, entre l'antenne et les capteurs (en mètres) ;
- **une latence d'horodatage** : le décalage d'horloge de la section 2 est calculé à partir de l'heure à laquelle le PC *reçoit* la trame GPS. La trame arrive un peu après l'instant qu'elle décrit, donc le décalage est surestimé et chaque mesure est datée un peu trop tôt. L'erreur en mètres vaut vitesse × latence.

**Idée.** Une rue parcourue dans les deux sens voit les mêmes luminaires aux mêmes endroits. Si les mesures sont décalées de d mètres vers l'avant, le profil du sens 1 est décalé de +d et celui du sens 2 de −d : les bulles de lumière des deux sens ne se superposent plus. On cherche donc le décalage qui maximise la **corrélation entre les deux sens**, case par case (1 m). C'est indépendant de l'inventaire.

**Séparer les deux erreurs.** On fait la recherche pour plusieurs classes de vitesse. Le décalage optimal suit une droite : décalage(v) = bras de levier + v × latence. La pente donne la latence, l'ordonnée à l'origine donne le bras de levier.

**Ormesson.**

| Vitesse médiane | Décalage optimal | Concordance aller/retour sans correction | avec correction |
|---|---|---|---|
| 3,8 m/s (14 km/h) | −1,2 m | 0,87 | 0,96 |
| 6,7 m/s (24 km/h) | −1,9 m | 0,72 | 0,94 |
| 9,1 m/s (33 km/h) | −2,9 m | 0,49 | 0,94 |

On en déduit une **latence de 0,33 s** et un **bras de levier de +0,2 m** : les capteurs sont pratiquement à l'aplomb de l'antenne. À 33 km/h, sans correction, chaque mesure tombait à près de 3 m de sa vraie place, dans un sens puis dans l'autre. C'était la principale cause d'imprécision sur la carte.

**Correction appliquée.** On ajoute la latence aux instants de mesure. Puis on place chaque capteur sur la **trace réelle de l'antenne**, à (distance parcourue à cet instant − bras de levier). En suivant la trace, et non une simple ligne droite, la remorque passe bien là où la voiture est passée, même dans les virages.

**Effet mesuré.** L'écart médian entre passages dans un même hexagone tombe de 17 % à 12 %.

**Réglage.** `--calage-auto` est actif par défaut et le calcul est refait à chaque tournée. On peut aussi imposer des valeurs mesurées sur la remorque avec `--decalage-long-m` et `--latence-s`.

## 3. Positions et vitesse

- Les coordonnées WGS84 sont converties en **Lambert 93 (EPSG:2154)** pour calculer en mètres.
- La vitesse et le cap sont calculés à partir des positions lissées sur 1 s (5 points à 5 Hz). Ils ne viennent pas des trames RMC, absentes du port A.
- Les positions sont **interpolées** à chacun des 616 000 instants de mesure. Si le GNSS a une coupure de plus de 2 s, on n'interpole pas à travers ce trou.

**Qualité GNSS observée :** 93 % des positions en DGPS (différentiel), 7 % en RTK flottant, HDOP médian de 0,5 et 14 satellites. Les positions sont donc précises au mètre près, ce qui suffit pour rattacher une mesure à un support.

---

## 4. Nettoyage du signal (`clean_lux`)

Le signal passe par cinq étapes, dans cet ordre.

### 4.1 Valeurs négatives ramenées à 0
**Constat.** Environ 1 % des échantillons sont négatifs, jusqu'à −4 lx.
**Déduction.** La propriété `lux_offset` de chaque capteur (3,8, 8,1 et 3,7 lx) a été soustraite à l'acquisition. Dans le noir complet, la valeur passe donc sous zéro. Un éclairement négatif n'a pas de sens physique, on le ramène à 0.

### 4.2 Parasites sur un seul capteur : filtre de Hampel
**Méthode.** Pour chaque capteur, on calcule sur une fenêtre glissante de 0,5 s :
- la médiane locale ;
- l'écart absolu médian (MAD), multiplié par 1,4826 pour être comparable à un écart-type.

Un échantillon est un parasite s'il s'écarte de la médiane de plus que le maximum de ces trois seuils : 8 lx, 4 × MAD, ou 60 % de la médiane. Il est alors remplacé par la médiane locale.
**Pourquoi.** Un parasite électrique ou un reflet bref touche un échantillon isolé. La médiane n'est pas influencée par ces valeurs extrêmes, contrairement à la moyenne.
**Résultat.** 0,01 % des échantillons sont corrigés. Les capteurs sont propres.

### 4.3 Combinaison des 3 capteurs par la médiane
**Méthode.** À chaque instant, l'éclairement retenu est la médiane des trois capteurs.
**Pourquoi.** Les trois capteurs sont espacés de 98 cm sur la remorque et mesurent presque la même chose (corrélations de 0,94 à 0,98). Si l'un d'eux est ébloui (phare, reflet latéral), la médiane l'ignore, alors que la moyenne serait tirée vers le haut.
**Contrôle de cohérence.** On compte les instants où l'écart entre capteurs dépasse 30 lx et vaut plus que la médiane (`ecart_capteurs_max`). Cela représente 0,02 % du temps.

### 4.4 Pics aberrants : valeur trop haute ou pic incohérent
C'est le filtrage demandé. Chaque pic du signal combiné est d'abord **mesuré** :
- sa hauteur ;
- sa proéminence, c'est-à-dire de combien il dépasse le niveau autour de lui (`scipy.signal.find_peaks`) ;
- sa **largeur à mi-hauteur** en secondes (`peak_widths`), **convertie en mètres** en la multipliant par la vitesse du véhicule.

Seuls les pics d'au moins 15 lx de proéminence qui **doublent au moins** le niveau ambiant sont examinés. Les ondulations normales du signal ne sont jamais touchées.

Un pic examiné est déclaré aberrant dans trois cas.

**a) Trop haut** (`lux_max_plausible` = 150 lx).
*Raisonnement.* Sur toute la tournée, 99 % des points sont sous 73 lx. Les luminaires d'éclairage public les plus puissants donnent moins de 100 lx à 1,5 m du sol, hauteur des capteurs. Une valeur au-dessus de 150 lx ne vient donc pas de l'éclairage public. Ce peut être un projecteur de stade ou d'enseigne, des phares en plein face, ou un défaut capteur.
*Ormesson.* 3 pics : 433 lx à 23h31, puis 249 et 209 lx à 23h32. Ils sont exclus des statistiques d'éclairage public. Si on veut les repérer comme sources, ils restent visibles dans la table « pics aberrants » du rapport.

**b) Trop bref, véhicule en mouvement** (largeur minimale déduite de la géométrie, 1,84 m).
*Raisonnement physique.* Les capteurs sont sur une remorque, à **1,5 m du sol** (`hauteur_capteurs_m`). Ce qui compte est donc la hauteur du feu **au-dessus des capteurs** : Δh = hauteur du feu − 1,5 m. Sous une source ponctuelle, l'éclairement horizontal suit E(x) ∝ Δh³ / (Δh² + x²)^(3/2). Il tombe à la moitié de son maximum pour x = 0,77 Δh, donc la largeur à mi-hauteur vaut **1,53 × Δh**.
Pour le feu le plus bas retenu en voirie (3 m, `hauteur_feu_min_m`), Δh = 1,5 m et la largeur vaut 2,3 m. On garde 20 % de marge, soit un seuil de **1,84 m**. Un pic plus étroit ne peut pas venir d'un luminaire : c'est une lumière transitoire (phare d'un véhicule croisé, flash, reflet sur une vitre). Les bornes basses, plus basses que les capteurs, ne produisent pas de cloche verticale et ne sont pas concernées.
*Ormesson.* Une quinzaine de pics, par exemple un pic à 17 lx qui dure moins de 0,2 s à 34 km/h dans une rue à 3 lx (rue des Martyrs de Châteaubriant, 21h31).

**c) Variation à l'arrêt.**
Si le véhicule roule à moins de 0,5 m/s, toute variation de lumière vient d'une source qui change (phares d'une voiture derrière, clignotant) et non d'un luminaire qu'on dépasse. Ces instants sont de toute façon exclus des statistiques (section 5). On les compte à part pour ne pas les confondre avec les pics trop brefs. *Ormesson :* 5 cas.

**Correction.** Pour chaque pic aberrant, toute sa base (de 5 % à 95 % de sa hauteur) est retirée. Elle est remplacée par une **interpolation linéaire** entre les valeurs valides de part et d'autre, sur le signal combiné comme sur chacun des 3 capteurs. On ne met pas zéro, ce qui créerait un faux « trou noir » sur la carte.

### 4.5 Passages incohérents (section 6)
Une lumière transitoire plus longue, par exemple un bus éclairé qui roule à côté, échappe au critère de largeur. On la détecte en comparant les passages entre eux. Dans une maille de 10 m parcourue **au moins 3 fois**, un passage est écarté si sa moyenne dépasse **3 fois la médiane des passages** et la dépasse aussi de plus de 20 lx (`ratio_passage_max`).
*Pourquoi au moins 3 passages.* Avec seulement 2 passages, on ne sait pas lequel est le bon.
*Ormesson.* 3 passages écartés. Leurs pics ne servent pas à estimer les lampadaires.

---

## 5. Exclusion des arrêts et ré-échantillonnage à pas de distance

**Arrêts.** Sous 0,5 m/s (`vitesse_min`), le véhicule est arrêté : feu rouge, carrefour, manœuvre. Ces mesures répètent le même point des centaines de fois. On les exclut, ce qui représente 18,6 % du temps à Ormesson.

**Pas de distance.** Le signal est regroupé en **un point tous les 2 m roulés** (`pas_m`), en faisant la moyenne des échantillons de chaque tranche de 2 m.
*Pourquoi.* En pondérant par le temps, une rue parcourue à 10 km/h compterait trois fois plus qu'une rue parcourue à 30 km/h. Avec un pas de distance, chaque mètre de voie pèse autant. C'est la réponse à la question du PDF : « quelle méthode est la plus représentative ».
*Effet mesuré.* La moyenne passe de 19,1 lx (pondération au temps) à 19,7 lx (pondération à la distance). Le rapport affiche les deux.

---

## 6. Sens de passage et répétabilité (sans hexagones ni filaire)

**Sens 1 / sens 2.** Sans filaire de voirie, l'axe de la voie est estimé localement, dans des mailles de 30 m. On fait la moyenne axiale des caps : on double les angles, pour que 30° et 210° comptent comme le même axe. Le sens 1 est orienté vers le nord, ou vers l'est pour une voie est-ouest. Un point est en sens 1 si son cap est à moins de 90° de cet axe, en sens 2 sinon. « Gauche » et « droite » d'un lampadaire sont toujours donnés par rapport au sens 1. Le lampadaire garde ainsi le même côté quel que soit le sens dans lequel la remorque est passée.

**Passages multiples.** Une maille interne de 10 m sert à comparer les passages au même endroit. Elle n'est ni affichée ni exportée. On l'utilise pour deux choses :
- écarter les passages incohérents (section 4.5) ;
- mesurer la répétabilité. À Ormesson, l'éclairement moyen d'une maille varie de **12 %** d'un passage à l'autre, en médiane.

Les hexagones ont été retirés du rapport et de la carte à la demande de l'utilisateur.

---

## 7. Lampadaires estimés à partir de l'éclairement (`estimer_lampadaires`)

**Pics.** Chaque passage sous un lampadaire crée une « cloche » dans l'éclairement filtré. On cherche les pics avec `find_peaks` sur le signal au pas de 2 m, lissé sur 3 points. Un pic doit dépasser son voisinage d'au moins 3 lx (`prominence_min_lx`) et d'au moins 30 % de sa hauteur, et les pics doivent être espacés d'au moins 8 m. *Ormesson : 2 089 pics.*

**Regroupement des passages.** Les pics de passages différents situés à moins de 6 m sur la trace (`rayon_regroupement_m`) correspondent au même lampadaire. On les regroupe en partant des pics les plus forts. *Ormesson : 1 270 lampadaires, dont 385 vus sur au moins deux passages.* L'interdistance médiane entre lampadaires voisins est de 24 m, ce qui est cohérent avec l'implantation usuelle en voirie (20 à 30 m).

**Position le long de la voie.** C'est le point de la trace où la cloche est maximale. Avec le calage GPS / remorque (section 2 bis), les passages concordent à 0,4 m près. L'écart médian avec l'inventaire est de 2,8 m, mais l'inventaire est lui-même imprécis de plusieurs mètres.

**Gauche ou droite.**
- *Principe.* Lux1 et Lux3 sont écartés de 0,98 m (propriété `lux_position` du TDMS : y = −0,49 et +0,49 m, y positif à gauche). Le capteur du côté du feu en est un peu plus près et reçoit donc plus de lumière.
- *Asymétrie.* On la calcule par `a = (gauche − droite) / (2 × proéminence du pic)`. On divise par la proéminence plutôt que par la valeur totale. Ainsi, la lumière de fond, qui vient des lampadaires voisins et frappe les deux capteurs à peu près également, ne dilue pas l'écart.
- *Vote.* L'asymétrie de chaque passage est ramenée au repère du sens 1, puis on fait la moyenne. Le lampadaire est classé à gauche ou à droite si cette moyenne dépasse 0,04 en valeur absolue (`seuil_asym_cote`) et si au moins 70 % des passages sont d'accord (`accord_cote_min`).
- *Ormesson.* 603 lampadaires à gauche, 453 à droite.

**Deux côtés.** Avec seulement 0,98 m entre les capteurs, trois situations donnent la même mesure, avec une asymétrie faible :
- un feu au-dessus de l'axe ;
- deux feux face à face ;
- un feu éloigné.

L'outil les classe « non tranchés » (214 à Ormesson). La comparaison avec l'inventaire montre qu'on ne peut pas faire mieux point par point. Sur 91 lampadaires vus sur plusieurs passages avec une asymétrie faible, un seul correspondait à deux supports face à face ; les autres étaient des feux isolés.

La question « un côté ou les deux » est donc tranchée **par tronçon de 100 m** (`troncon_m`), à partir des lampadaires dont le côté est connu :
- **les deux côtés** (implantation bilatérale ou en quinconce) si le côté minoritaire représente au moins 25 % des lampadaires du tronçon ;
- **un seul côté** sinon.

*Ormesson : 75 tronçons ont des lampadaires des deux côtés et 146 d'un seul côté.*

**Distance au bord de la voie.** On a essayé de la déduire de la physique. Pour une source ponctuelle, la largeur à mi-hauteur de la cloche vaut 1,53 × la distance au feu, et l'asymétrie vaut 1,47 × d / R². L'estimation obtenue s'est révélée **moins bonne qu'une valeur fixe** : erreur médiane de 2,1 m contre 1,3 m. En effet, les optiques routières ne sont pas des sources ponctuelles, et la lumière des voisins déforme la cloche. Le lampadaire est donc placé à **5,3 m** de la trace (`dist_laterale_m`), distance médiane observée. Dans la moitié des cas, la distance réelle est entre 4,1 et 6,7 m. La hauteur de feu n'est pas estimée, pour la même raison.

---

## 8. Type de lampe (chromamètre)

Le chromamètre mesure environ une fois par seconde l'éclairement, la température de couleur proximale (Tcp) et l'écart à la courbe des corps noirs (deltaUV). Pour chaque pic, on garde ses mesures à ± 1,5 s du maximum, avec un éclairement supérieur à 3 lx, puis on prend la médiane sur tous les passages du lampadaire.

| Classe | Règle | Ce que c'est |
|---|---|---|
| Sodium haute pression | Tcp = 0 dans au moins 50 % des mesures | Le chromamètre renvoie 0 sous ~2 200 K ; le sodium est à ~2 000 K. |
| Blanc chaud 2 700 K | Tcp < 2 800 K | LED 2 700 K **ou** iodures métalliques céramiques (CosmoWhite) : même Tcp, indiscernables |
| LED 3 000 K | 2 800 à 3 300 K | presque toujours des LED |
| Blanc neutre | 3 300 à 3 800 K | vapeur de mercure, iodures, LED 3 500 K |
| Blanc froid | > 3 800 K | LED 4 000 K ou iodures |

Le deltaUV a été regardé mais ne sépare pas mieux les familles. Seules les iodures métalliques classiques ont un deltaUV nettement plus élevé (0,012), et il n'y en a que quelques-unes. La fréquence d'échantillonnage des luxmètres (50 Hz) ne permet pas non plus de voir le scintillement à 100 Hz des lampes à décharge, qui aurait distingué une lampe à décharge d'une LED.

---

## 9. Puissance et consommation

**Constat.** L'éclairement au passage ne dit presque rien de la puissance. Sur Ormesson, la corrélation entre log(proéminence du pic) et log(puissance déclarée) n'est que de **0,08**. La hauteur, l'optique, la distance du feu, l'encrassement et le vieillissement pèsent bien plus que les watts. Une formule « lux → watts » donnerait donc des puissances fausses, du simple au triple.

**Méthode retenue.** Chaque lampadaire reçoit la **puissance moyenne de son type**, avec la fourchette interquartile observée (`PUISSANCE_TYPE`) :

| Type | Puissance retenue | Fourchette |
|---|---|---|
| Sodium HP | 108 W | 100–150 W |
| Blanc chaud 2 700 K | 104 W | 90–140 W |
| LED 3 000 K | 69 W | 45–90 W |
| Blanc neutre | 78 W | 40–100 W |
| Blanc froid | 122 W | 100–140 W |

**Consommation.** Les lampes à décharge ont des pertes de ballast : +12 % pour le sodium, +5 à 10 % pour les autres. Pour les LED, l'alimentation est déjà incluse dans la puissance. Le calcul annuel suppose **4 100 h** d'allumage, la durée usuelle de l'éclairage public en France.

**Ormesson.** 127 kW installés (fourchette 111 à 174 kW), 140 kW consommés, environ 574 MWh par an. Le sodium représente 95 kW consommés.

**Limite.** Ces puissances moyennes ont été étalonnées sur Ormesson. Pour une commune au parc différent (LED récentes de 20 à 40 W, par exemple), il faudra les ajuster.

---

## 10. Précision des estimations (contrôle a posteriori avec l'inventaire d'Ormesson)

L'outil n'utilise pas l'inventaire. Pour savoir ce que valent les estimations, on les a comparées une fois à l'Excel d'Ormesson. Chaque lampadaire estimé est apparié à au plus un support, par affectation optimale à 12 m près. Les supports retenus sont ceux à moins de 10 m de la trace, et l'inventaire est translaté de 2 m pour corriger son décalage d'ensemble.

| Estimation | Précision constatée |
|---|---|
| Nombre de lampadaires | 1 270 estimés pour 1 264 supports à moins de 10 m de la trace ; 72 % tombent sur un support |
| Position le long de la voie | écart médian de 2,8 m (l'inventaire est lui-même faux de plusieurs mètres pour ~170 supports) |
| Côté gauche / droite | juste dans 90 % des cas (737 comparés) |
| Un côté ou les deux, par tronçon de 100 m | juste pour 80 % des tronçons (195 comparés) |
| Sodium ou source blanche | juste dans 84 % des cas (luminaires non modifiés depuis le relevé) |
| « LED 3 000 K » | 84 % sont bien des LED |
| « Blanc chaud 2 700 K » | 68 % de CosmoWhite, 27 % de LED |
| Puissance d'un lampadaire | écart médian de 20 % (75 % des cas sous 29 %) |
| Puissance totale | 63,6 kW estimés pour 63,5 kW à l'inventaire sur les mêmes supports. L'étalonnage a été fait sur ces données : à recontrôler dans une autre commune. |

Ces chiffres sont repris dans le rapport (constante `VALIDATION`), pour que le lecteur sache quelle confiance accorder à chaque estimation.

---

## 11. Abaissement nocturne

On compare le pic d'un **même lampadaire** vu avant et après 22h (`heure_abaissement`). Une baisse de plus de 30 % le marque comme abaissé. À Ormesson, seuls 34 lampadaires ont été repassés après 22h. C'est trop peu pour conclure, et le rapport le dit. Pour mesurer l'abaissement, il faut prévoir de repasser après 22h dans des rues déjà parcourues.

---

## 12. Rapport et exports

- `rapport.html` est autonome : les données y sont intégrées en JSON et la carte est dessinée en canvas, zoomable. Il fonctionne sans serveur. Seules les polices viennent de Google Fonts, avec repli sur les polices système.
- `lampadaires_estimes.csv` et `lampadaires_estimes.geojson` contiennent un lampadaire par ligne. Pour chacun, on trouve :
  - la position (WGS84 et Lambert 93) et le point de la trace au pic ;
  - le côté et l'implantation du tronçon ;
  - le nombre de passages, l'éclairement au pic, l'asymétrie et l'accord entre passages ;
  - le type de lampe, la Tcp et le deltaUV ;
  - la puissance, sa fourchette et la consommation ;
  - le pic avant et après 22h.
- `mesures_pas2m.csv` contient un point tous les 2 m : les 3 capteurs, la médiane, la vitesse, le cap, le sens et l'heure.
- Les CSV utilisent le séparateur `;` et la virgule décimale, pour Excel en français. Les GeoJSON sont en WGS84 et s'ouvrent dans QGIS.
- `resume.json` contient tous les indicateurs, les paramètres et le temps de chaque étape.
- Un rapport existant n'est jamais écrasé : il est déplacé dans `rapports/archives/`.

---

## 12 bis. Carte satellite (`carte.html`)

La carte utilise **Leaflet** avec l'imagerie satellite **Esri World Imagery**, visuellement équivalente à Google Maps (qui exige une clé d'API payante). Un fond « plan » OpenStreetMap est aussi proposé.

**Calques :**
- **Points chauds / froids** : un dégradé lissé. Les mesures sont moyennées dans une grille à l'écran, puis chaque case est dessinée en disque flouté. Le bleu-vert sombre correspond à 0 lx et l'orange à 50 lx ou plus.
- **Capteurs filtrés** : gauche, centre et droite, plus la médiane. Ils sont dessinés à 2 m de part et d'autre de la trace pour être lisibles ; leur écart réel est de 0,49 m.
- **Sens 1 / sens 2** : filtre qui s'applique aux points chauds / froids et aux capteurs.
- **Lampadaires estimés** : colorés par type de lampe, ou par côté (case « couleur = côté »). Un contour pointillé signale un côté non tranché. Le survol donne le type, la Tcp, la puissance et sa fourchette, le nombre de passages, l'abaissement éventuel et l'implantation du tronçon.
- **Tronçons à lampadaires des deux côtés** : un halo vert sous les lampadaires de ces tronçons.

**Calage sur l'image satellite.** Les flèches décalent toutes les couches par pas de 0,5 m, pour absorber l'erreur de géoréférencement de l'imagerie (1 à 3 m). Le réglage est mémorisé, et la commande pour l'appliquer aux prochains rapports est affichée.

---

## 13. Hypothèses à confirmer

1. **Puissances moyennes par type** étalonnées sur Ormesson. À ajuster pour un autre parc.
2. **Distance latérale fixe de 5,3 m.** C'est une valeur typique, pas une mesure.
3. **Seuil de 10 lx** comme repère, faute de classes EN 13201 par voie.
4. **4 100 h d'allumage par an** pour l'énergie.
5. **150 lx comme plafond de plausibilité.** À relever si des zones très éclairées (places, stades) font partie du périmètre.
6. **Mesure sur remorque, capteurs à 1,5 m du sol**, et non au sol selon la grille normative.
