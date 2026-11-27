"""Import des seuils SDAL depuis un fichier (.csv, séparateur ; ou ,).

Deux formats, reconnus par leurs colonnes :
  * par type de voie : colonnes « type » (ou « classe ») et « seuil »
      type;seuil
      Voie principale;15
      residentielle;7,5
  * par rue : colonnes « rue » (ou « voie », « nom ») et « seuil » ; le seuil d'une rue remplace celui de son type
      rue;seuil
      Rue du Général de Gaulle;20
Les noms sont comparés sans accents ni majuscules.
"""
from __future__ import annotations

import base64
import io
import re
import unicodedata

import pandas as pd

COL_TYPE = ("type", "classe", "type de voie", "type_de_voie", "categorie")
COL_RUE = ("rue", "voie", "nom", "nom de la voie", "nom_voie", "adresse")
COL_SEUIL = ("seuil", "seuil_lx", "seuil (lx)", "emoy", "e_moy", "lux", "eclairement", "valeur")


def normaliser(texte) -> str:
    s = unicodedata.normalize("NFD", str(texte or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _colonne(df, candidats):
    noms = {normaliser(c): c for c in df.columns}
    return next((noms[normaliser(c)] for c in candidats if normaliser(c) in noms), None)


def lire(contenu_b64: str, classes: dict) -> dict:
    """contenu (data URL de dcc.Upload) -> {"types": {classe: seuil}, "rues": {nom normalisé: seuil},
    "message": str}. classes : troncons.CLASSES (id -> (libellé, highways, seuil par défaut))."""
    brut = base64.b64decode(contenu_b64.split(",", 1)[-1])
    texte = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            texte = brut.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if texte is None:
        raise ValueError("Fichier illisible : enregistrez-le en CSV (UTF-8).")
    sep = ";" if texte.count(";") >= texte.count(",") else ","
    df = pd.read_csv(io.StringIO(texte), sep=sep, dtype=str).dropna(how="all")
    c_seuil = _colonne(df, COL_SEUIL)
    if c_seuil is None:
        raise ValueError("Colonne « seuil » introuvable.")
    seuils = pd.to_numeric(df[c_seuil].str.replace(",", ".", regex=False).str.strip(), errors="coerce")
    c_rue, c_type = _colonne(df, COL_RUE), _colonne(df, COL_TYPE)
    if c_rue is not None:
        rues = {normaliser(n): float(s) for n, s in zip(df[c_rue], seuils) if normaliser(n) and pd.notna(s)}
        if not rues:
            raise ValueError("Aucune rue avec un seuil valide.")
        return {"types": {}, "rues": rues, "message": f"{len(rues)} rue(s) avec un seuil propre."}
    if c_type is None:
        raise ValueError("Colonne « type » ou « rue » introuvable.")
    alias = {}
    for cid, (lib, highways, _) in classes.items():
        for a in (cid, lib, *highways, lib.replace("Voie ", "")):
            alias[normaliser(a)] = cid
    types, inconnus = {}, []
    for n, s in zip(df[c_type], seuils):
        cid = alias.get(normaliser(n))
        if cid and pd.notna(s):
            types[cid] = float(s)
        elif normaliser(n):
            inconnus.append(str(n))
    if not types:
        raise ValueError("Aucun type de voie reconnu (principale, distribution, résidentielle, desserte).")
    msg = f"{len(types)} type(s) de voie importé(s)."
    if inconnus:
        msg += f" Ignoré(s) : {', '.join(inconnus[:5])}."
    return {"types": types, "rues": {}, "message": msg}
