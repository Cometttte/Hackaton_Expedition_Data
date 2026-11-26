# Explication du code : outil de rapport de fin de tournée GETYS

Ce document décrit le fonctionnement de l'outil étape par étape. Pour chaque déduction, il indique ce qu'on en conclut, sur quoi la conclusion repose (physique, données observées) et le paramètre qui la règle.
Les chiffres d'exemple viennent de la tournée d'Ormesson-sur-Marne du 18/03/2025.

---

## 0. Organisation des fichiers

| Fichier | Rôle |
|---|---|
| `getys_rapport.py` | Point d'entrée en ligne de commande. Il lance l'analyse, écrit le rapport HTML et peut surveiller un dossier pour traiter chaque nouvelle tournée. |
| `getys_analyse.py` | Toute l'analyse. La fonction `analyser()` enchaîne les étapes décrites ci-dessous et renvoie un dictionnaire de résultats. |
| `rapport_template.html` | Gabarit du rapport. Le script y injecte les résultats en JSON. Les textes du rapport sont générés à partir des chiffres, donc ils s'adaptent à chaque tournée. |
| `Params` (dans `getys_analyse.py`) | Tous les seuils au même endroit. Chacun se règle en ligne de commande (`--seuil-eteint-lx 3`, par exemple). |

Durée totale sur la tournée d'exemple (3h20, 72 km) : environ 5 secondes. Les calculs sont vectorisés avec numpy et scipy, sans boucle Python sur les 616 000 échantillons. La seule boucle porte sur les 1 538 supports.

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

### 4.5 Passages incohérents (au niveau des hexagones, section 6)
Une lumière transitoire plus longue, par exemple un bus éclairé qui roule à côté, échappe au critère de largeur. On la détecte en comparant les passages entre eux. Dans un hexagone parcouru **au moins 3 fois**, un passage est écarté si sa moyenne dépasse **3 fois la médiane des passages** et la dépasse aussi de plus de 20 lx (`ratio_passage_max`).
*Pourquoi au moins 3 passages.* Avec seulement 2 passages, on ne sait pas lequel est le bon.
*Ormesson.* 2 passages écartés.

---

## 5. Exclusion des arrêts et ré-échantillonnage à pas de distance

**Arrêts.** Sous 0,5 m/s (`vitesse_min`), le véhicule est arrêté : feu rouge, carrefour, manœuvre. Ces mesures répètent le même point des centaines de fois. On les exclut, ce qui représente 18,6 % du temps à Ormesson.

**Pas de distance.** Le signal est regroupé en **un point tous les 2 m roulés** (`pas_m`), en faisant la moyenne des échantillons de chaque tranche de 2 m.
*Pourquoi.* En pondérant par le temps, une rue parcourue à 10 km/h compterait trois fois plus qu'une rue parcourue à 30 km/h. Avec un pas de distance, chaque mètre de voie pèse autant. C'est la réponse à la question du PDF : « quelle méthode est la plus représentative ».
*Effet mesuré.* La moyenne passe de 19,1 lx (pondération au temps) à 19,7 lx (pondération à la distance). Le rapport affiche les deux.

---

## 6. Hexagones et multi-passages

**Grille.** Ce sont des hexagones « pointe en haut » de 10 m de rayon (`hex_m`). Chaque point de 2 m est rangé dans l'hexagone qui le contient, avec un arrondi en coordonnées cubiques (`hex_index`).

**Passages.** Dans un même hexagone, deux points séparés de plus de 20 s (`gap_passage_s`) appartiennent à deux passages différents.

**Statistiques par hexagone.**
- `E_moy` : moyenne de chaque passage, puis moyenne des passages. Une rue parcourue 10 fois ne pèse pas 10 fois plus. C'est la gestion des multi-passages demandée dans le PDF.
- `E_med`, `E_p10`, `E_min`, `E_max` : calculés sur tous les points.
- `U0 = P10 / moyenne` : uniformité. On prend le 10e centile plutôt que le minimum, qui est trop sensible à un seul point sombre.
- `cv_passages` : écart-type divisé par la moyenne entre les passages. Il mesure la répétabilité. *Ormesson :* 12 % en médiane après le calage de la section 2 bis (17 % avant).
- `conforme` : `E_moy` ≥ 10 lx (`seuil_emoy_lx`). C'est un **seuil provisoire** en attendant le SDAL.

---

## 7. Inventaire Excel (`load_inventaire`)

- On lit l'onglet `SUPPORT ECLAIRAGE PUBLIC`, avec l'en-tête sur la 2e ligne. Les autres onglets décrivent le réseau (armoires, câbles).
- **Déduction : une ligne par luminaire, pas par support.** Il y a 1 624 lignes pour 1 538 `ID` distincts, parce qu'un support double ou triple occupe plusieurs lignes. On regroupe donc par `ID` et on additionne les puissances.
- **Déduction : l'inventaire est postérieur au relevé.** 454 luminaires ont une date de pose ou de relampage après le 18/03/2025. Ils sont marqués `modif_apres_releve` et ne sont pas comptés comme des erreurs d'inventaire.
- Les coordonnées latitude et longitude sont converties en Lambert 93. On ne réutilise pas les champs X/Y de l'Excel, qui sont en CC49 (EPSG:3949).

---

## 8. Supports : maximum au passage et supports suspects

Pour chaque support :
1. On cherche les points mesurés à moins de 12 m (`rayon_support_m`), avec un arbre KD (`cKDTree`).
2. On sépare les passages (écart de plus de 20 s) et on garde le **maximum de chaque passage**. Le maximum correspond au moment où le véhicule passe sous le luminaire.
3. On retient la **médiane des maxima** des passages. Elle résiste mieux qu'un maximum global à un passage isolé perturbé.
4. Le support est **« frôlé »** si la trajectoire passe à moins de 10 m (`dist_frole_m`). Au-delà, l'absence de lumière ne prouve rien.
5. Le support est **suspect** s'il est frôlé et que son maximum au passage reste **sous 5 lx** (`seuil_eteint_lx`). Un luminaire allumé à moins de 10 m donne toujours plus de 10 lx sur les capteurs à 1,5 m du sol (le 10e centile des LED est à 14 lx). Sous 5 lx, il est éteint ou très dégradé.

**Armoires.** Si une armoire (colonne `PARENT`) a au moins 3 supports suspects et au moins 30 % de ses supports frôlés suspects, on soupçonne un circuit coupé plutôt que des pannes isolées.

**Rues.** Chaque point de mesure est rattaché à la rue (`ZONE`) du support le plus proche, à moins de 30 m. On en déduit la moyenne, le P10, l'uniformité et le nombre de passages par rue. Cela remplace la « moyenne au tronçon », faute de filaire de voirie.

---

## 9. Abaissement nocturne de 22h

**Problème.** Comparer tout ce qui est mesuré avant 22h à tout ce qui est mesuré après ne compare pas les mêmes rues.
**Méthode appariée.** On garde uniquement les hexagones parcourus **avant et après** 22h. Pour chacun, on calcule le rapport « moyenne après / moyenne avant », puis on prend la médiane, en séparant les hexagones proches (moins de 25 m) d'un luminaire en régime « Variation » des autres.
**Attendu.** Environ 50 % d'après l'inventaire (« de 22h à 24h 50% »).
**Mesuré à Ormesson.** 96 % sur 51 hexagones, donc pas de baisse visible. Soit l'abaissement n'était pas actif le soir du relevé, soit l'horaire diffère.

---

## 10. Chromamètre : type de source (`Calorimetre`)

**Constat dans les données.** La température de couleur (Tcp) vaut **0 dans 70 % des mesures**, même quand la lumière est forte. Aucune valeur non nulle n'est sous 2 200 K.
**Déduction.** Sous environ 2 200 K, l'appareil ne sait pas calculer la Tcp et renvoie 0. Le sodium haute pression émet vers 2 000 K. **Tcp = 0 avec Ev > 5 lx signifie donc « source sodium ou ambrée ».**
**Vérification sur les données.**
- 94 % des supports déclarés sodium (non modifiés depuis le relevé) donnent bien Tcp = 0.
- **94 % des LED posées après le relevé donnent Tcp = 0.** Le soir du relevé, c'était encore du sodium. Cela confirme la règle et montre que l'outil peut dater un remplacement.

**Règle.** Au passage d'un support (± 2 s autour de son maximum), si la majorité des mesures Tcp valent 0, la source est classée « sodium ». Sinon elle est « blanche », avec la Tcp médiane mesurée.
**Écarts signalés.** Un écart n'est signalé que si tous les échantillons du passage concordent, si le support n'a pas été modifié depuis et s'il n'est pas déclaré à 2 200 K (LED ambrée).
**Limite.** Une LED isolée entre des lampes sodium est souvent masquée par ses voisines : les LED en place ne sont vues « blanches » que dans 66 % des cas. Le classement est indicatif.

---

## 11. Bonus : détection des luminaires et côté gauche ou droite

**Détection.** Chaque luminaire dépassé crée une « cloche » dans le signal. On cherche les pics avec une proéminence d'au moins 3 lx et d'au moins 30 % de la hauteur du pic, espacés d'au moins 8 m (4 points de 2 m).

**Côté.** Lux1 et Lux3 sont à ±0,49 m de l'axe du véhicule. Le capteur placé du côté du luminaire reçoit un peu plus de lumière. L'asymétrie `(L1 − L3) / (L1 + L3)` donne donc le côté.

**Calibration automatique.** Les propriétés `lux_position` ne disent pas quel signe correspond à la gauche. L'outil le déduit des données. Pour chaque support frôlé, on connaît son côté géométrique (produit vectoriel entre le cap du véhicule et la direction du support) et le signe de l'asymétrie mesurée. Le signe le plus fréquent fixe la convention.
*Ormesson.* Lux1 est **à droite**, ce qui est cohérent avec y = −0,49 dans le repère véhicule usuel (x vers l'avant, y vers la gauche). Le côté est juste dans **84 %** des cas.

**Position estimée.** On décale le point du pic de 3,5 m (`decalage_lateral_m`) vers le côté détecté. Les détections des différents passages sont regroupées à 8 m près, en traitant d'abord les pics les plus forts.

**Résultats.**
- 80 % des supports frôlés et allumés ont leur propre pic. Chaque pic n'est attribué qu'à un seul support (section 11 bis). Avant ce changement, on comptait un support comme détecté dès qu'un pic se trouvait à moins de 15 m, même s'il s'agissait de la lumière du voisin : le chiffre de 88 % était donc flatteur.
- 76 pics vus sur au moins 2 passages, au-dessus de 10 lx, attribués à aucun support et à plus de 7,5 m de tout ouvrage (supports recalés, éclairage sportif, mobilier urbain alimenté), correspondent à des **sources absentes de l'inventaire** : éclairage privé, vitrine ou support oublié.

---

## 11 bis. Recalage des positions de l'inventaire (`recaler_inventaire`)

**Constat.** Une vérification sur Google Street View a montré que les coordonnées de certains lampadaires dans l'Excel ne correspondent pas à leur emplacement réel. Si on mesure « autour » d'une position fausse, on attribue à un support la lumière de son voisin, ou on le déclare éteint à tort.

**Ce qu'on sait mesurer.** Quand la remorque passe au droit d'un luminaire, l'éclairement passe par un maximum. Après le calage GPS / capteurs (section 2 bis), la **position de ce maximum le long de la voie** est connue à quelques dizaines de centimètres près : les passages concordent avec un écart médian de 0,4 m. En revanche, la **distance latérale** (bord de chaussée, trottoir) ne se mesure pas avec trois luxmètres écartés de 0,49 m ; on ne connaît que le côté (gauche ou droite).

**Étapes.**
1. *Événements.* Chaque pic détecté à un passage (section 11) donne la position du véhicule au pic, sa direction de marche, le côté de la lumière et le groupe (luminaire) auquel il appartient.
2. *Couples candidats.* Pour chaque support et chaque luminaire détecté à proximité, on calcule à chaque passage la distance le long de la voie entre le support de l'Excel et le pic, et la distance latérale du support à la trajectoire. On garde les couples à moins de 15 m le long de la voie (`rayon_match_m`) et à moins de 15 m en latéral. Les passages en sens inverse sont remis dans le même repère, puis on prend la médiane.
3. *Affectation un pour un (algorithme hongrois, `linear_sum_assignment`).* Coût d'un couple = |décalage le long de la voie| + 4 m × (part des passages où le côté mesuré contredit le côté de l'Excel). Chaque luminaire va au support qui l'explique le mieux, et chaque support reçoit au plus un luminaire. C'est ce qui empêche un support mal placé de « voler » la lumière de son voisin.
4. *Décalage d'ensemble.* Sur les couples fiables, on résout par moindres carrés robustes (Huber, seuil 2 m) la translation Est / Nord qui explique au mieux tous les décalages le long des voies. Comme les rues ont des orientations variées, les deux composantes sont déterminées. **Ormesson : l'Excel est décalé de 2,0 m (1,7 m vers l'ouest, 1,1 m vers le sud).** Au-delà de 1 m, ce décalage est retiré de tout l'inventaire, puis l'appariement est refait. Il ne reste alors que 0,5 m de décalage d'ensemble.
5. *Position corrigée.* Le support est déplacé le long de la voie jusqu'au pic. Il est basculé de l'autre côté de la trajectoire seulement si au moins 75 % des passages voient la lumière du côté opposé et s'il est en bord de voie (≤ 8 m). Plus loin, la lumière vue peut venir d'une autre source.
6. *Fiabilité.* Un recalage est **fiable** si le luminaire est vu sur au moins 2 passages qui concordent à 2 m près (`dispersion_pics_max_m`). Un support fiable dont l'écart total à l'Excel dépasse **5 m** (`ecart_position_min_m`) est signalé **« position à corriger »**. Le seuil de 5 m vient de la somme de trois incertitudes : environ 1 m pour le GPS, environ 1 m de résidu de calage, et 2 à 3 m parce qu'un luminaire n'éclaire pas forcément exactement à l'aplomb du mât (crosse, inclinaison, optique).
7. *Mesures refaites.* Toutes les mesures par support (maximum au passage, frôlé ou non, suspect, côté, chromamètre) sont recalculées à la position corrigée. Les supports sans pic gardent la position de l'Excel, translatée du décalage d'ensemble.

**Résultats à Ormesson.**
- 996 supports ont été appariés à un point lumineux, dont 435 de façon fiable.
- L'écart médian entre l'Excel et la mesure est de 3,7 m. Le 90e centile est de 11,7 m.
- **168 supports sont à corriger**, dont 26 placés du mauvais côté de la rue. Leurs écarts restent sous la moitié de l'espacement entre supports voisins dans 90 % des cas, ce qui écarte une confusion systématique avec le voisin.
- Les supports suspects passent de 18 à 14 : 6 sortent de la liste et 2 y entrent. Il s'agit surtout de cas limites. Le décalage d'ensemble de 2 m les fait passer de l'autre côté d'un seuil : maximum entre 4 et 6 lx, ou distance au tracé autour de 10 m. Un seul, OR14030, sort de la liste parce que sa position a été recalée de 9 m.
- Validation par le chromamètre : 97 % des LED posées après le relevé sont vues en sodium, contre 95 % avant le recalage. Les mesures tombent davantage sur le bon luminaire.

**Sorties.**
- `positions_a_corriger.csv` liste chaque support à corriger avec sa position dans l'Excel, sa position mesurée, l'écart, le nombre de passages et la dispersion. Il est prêt pour la mise à jour du SIG.
- Les mêmes colonnes figurent dans `supports.csv` pour tous les supports.
- Dans le rapport, une section dédiée donne pour chaque support un lien Street View vers la position mesurée.
- Sur la carte satellite, le calque « Positions de l'inventaire à corriger » relie la position de l'Excel (rond gris) à la position mesurée (rond violet).

**Limite.** Si un support réel manque dans l'Excel, son pic peut être attribué au support inventorié le plus proche. Le recalage proposé est alors faux, avec un écart proche de 15 m. C'est pourquoi les écarts les plus grands sont en tête du tableau, à vérifier en priorité sur Street View.

---

## 12. Rapport et exports

- `rapport.html` est autonome : les données y sont intégrées en JSON et la carte est dessinée en canvas, zoomable. Il fonctionne sans serveur. Seules les polices viennent de Google Fonts, avec repli sur les polices système.
- Les exports CSV utilisent le séparateur `;` et la virgule décimale, pour s'ouvrir directement dans Excel en français. Les GeoJSON sont en WGS84 et s'ouvrent dans QGIS ou FME.
- `resume.json` contient tous les indicateurs, les paramètres utilisés et le temps de chaque étape, ce qui permet de suivre l'évolution d'une tournée à l'autre.

---

## 12 bis. Carte satellite (`carte.html`)

La carte est générée à partir de `carte_template.html` par `getys_rapport.py`. Elle utilise **Leaflet**, une bibliothèque de cartes web, avec comme fond l'imagerie satellite **Esri World Imagery**. Ce fond est équivalent visuellement à Google Maps. Google Maps n'a pas été retenu parce qu'il exige une clé d'API payante pour être intégré dans une page. Un fond « plan » OpenStreetMap est aussi proposé.

**Calques proposés :**
- **Points chauds / froids** : le dégradé lissé de l'exemple fourni.
  *Méthode.* On moyenne les mesures dans une grille à l'écran : c'est une valeur moyenne, pas une densité de points, sinon les rues parcourues plusieurs fois paraîtraient plus « chaudes ». On dessine ensuite chaque case, de la plus froide à la plus chaude, sous forme de disque d'environ 9 m de rayon. Enfin, on applique un flou gaussien.
  *Couleurs.* Le bleu-vert sombre correspond à 0 lx, le vert clair à 15 lx, le jaune à 22-30 lx et l'orange à 50 lx ou plus. On voit ainsi une bulle chaude sous chaque luminaire et un creux froid entre deux luminaires.
- **Capteurs filtrés** : gauche, centre et droite, sélectionnables séparément. Les valeurs viennent des 3 luxmètres après tout le filtrage de la section 4, avec un point tous les 2 m. Pour qu'on puisse les distinguer, les capteurs gauche et droit sont dessinés à 2 m de part et d'autre de la trace, perpendiculairement au cap du véhicule. Leur écart réel n'est que de 0,49 m.
- **Sens 1 / sens 2** : ce filtre s'applique aux points chauds / froids et aux capteurs.
  *Méthode.* Pour chaque rue, on calcule l'axe principal à partir de la moyenne axiale des caps : on double les angles, ce qui rend équivalents 30° et 210°. Un point est en sens 1 si son cap est à moins de 90° de cet axe, en sens 2 sinon. Les deux sens sont ainsi cohérents rue par rue, quelle que soit l'orientation de la rue.
- **Hexagones** : éclairement moyen par hexagone (même échelle de couleur), ou conformité au seuil (vert ou rouge).
- **Ouvrages** : supports d'éclairage public colorés par famille de source, supports suspects éteints (cercle rouge) et sources non inventoriées (cercle pointillé jaune).

**Calage sur l'image satellite.** Les mesures sont déjà corrigées de la latence et du bras de levier (section 2 bis). Il reste un écart possible : le géoréférencement de l'imagerie satellite elle-même, de 1 à 3 m selon les zones. Le panneau propose donc des flèches qui décalent toutes les couches par pas de 0,5 m. Le réglage est mémorisé dans le navigateur. La commande à ajouter pour que les prochains rapports s'ouvrent directement calés est affichée (`--recalage-est-m`, `--recalage-nord-m`).

**Performance.** Les quelque 36 000 points × 3 capteurs sont dessinés sur un canvas, et non comme 100 000 objets Leaflet séparés. La carte reste donc fluide. Le survol affiche les 3 valeurs de capteur, la médiane, le sens, le cap et l'heure.

## 13. Hypothèses à confirmer

1. **Pas de filaire de voirie** dans les entrées. Les moyennes sont calculées par rue, à partir des supports.
2. **Seuil de conformité unique de 10 lx**, faute de SDAL et de classes EN 13201 par voie.
3. **Hexagones de 10 m.** La taille utilisée dans le traitement actuel GETYS n'est pas connue.
4. **150 lx comme plafond de plausibilité.** À relever si des zones très éclairées (places, stades) font partie du périmètre.
5. **Mesure sur remorque, capteurs à 1,5 m du sol**, et non au sol selon la grille normative.
6. **Décalage d'affichage sur l'image satellite.** L'imagerie a sa propre précision de géoréférencement, de l'ordre de 1 à 3 m. Le réglage manuel de `carte.html` sert à absorber cet écart.
