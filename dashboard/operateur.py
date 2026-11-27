"""Vue opérateur : contrôle d'une tournée à partir du .tdms brut, SANS analyse (≈ 2 s).

  * trace des 3 luxmètres BRUTS sur la carte, côte à côte : pour chaque position GPS, la valeur de chaque
    capteur à cet instant, telle qu'enregistrée (ni moyenne, ni lissage, ni filtre, ni correction) ;
  * contrôles automatiques : trous GNSS, qualité du fix, capteur figé ou négatif, capteurs en désaccord ;
  * annotations : points posés sur la carte avec un commentaire, et notes libres de la tournée, enregistrés à
    côté du .tdms (<fichier>.notes.json) puis repris dans le rapport du chargé d'affaires.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))                  # getys_analyse.py (lecture du TDMS)
RECUS = Path(os.environ.get("TDMS_RECUS_DIR", RACINE / "tdms_recus"))
HEURE_LOCALE = ZoneInfo("Europe/Paris")
CAPTEURS = {"L1": "L1 (droite)", "L2": "L2 (centre)", "L3": "L3 (gauche)"}


def lister_tdms() -> list[Path]:
    """Fichiers .tdms reçus (tdms_recus/) et à la racine du dépôt, du plus récent au plus ancien."""
    fichiers = {}
    for d in (RECUS, RACINE):
        if d.exists():
            for f in d.glob("*.tdms"):
                fichiers.setdefault(f.name, f)
    return sorted(fichiers.values(), key=lambda f: f.stat().st_mtime, reverse=True)


def chemin_tdms(nom: str) -> Path | None:
    return next((f for f in lister_tdms() if f.name == nom), None)


# ------------------------------------------------------------------ LECTURE BRUTE
def lire_brut(chemin: Path) -> dict:
    return _lire_brut(str(chemin), chemin.stat().st_mtime)


@lru_cache(maxsize=3)
def _lire_brut(chemin: str, _mtime: float) -> dict:
    from nptdms import TdmsFile
    import getys_analyse as G

    t0 = time.time()
    tdms = TdmsFile.read(chemin)
    gnss, decalage, source, _ = G.load_gnss(tdms, lambda *_: None)
    L, t_pc, dt, meta = G.load_lux(tdms)
    t_lux = t_pc - decalage                      # horloge du PC ramenée sur l'heure GPS : sert à situer les mesures
    g = gnss[(gnss["lat"].abs() > 1e-6) & (gnss["lon"].abs() > 1e-6)]
    t = g["t"].to_numpy()
    # pour chaque position GPS : l'échantillon brut des luxmètres le plus proche dans le temps (aucun calcul dessus)
    i = np.round((t - t_lux[0]) / dt).astype(np.int64)
    ok = (i >= 0) & (i < L.shape[1])
    g, t, i = g[ok], t[ok], i[ok]
    d = pd.DataFrame({"t": t, "lat": g["lat"].to_numpy(), "lon": g["lon"].to_numpy(), "v": g["v"].to_numpy(),
                      "cap": g["cap"].to_numpy(), "L1": L[0, i], "L2": L[1, i], "L3": L[2, i]})
    med = d[["L1", "L2", "L3"]].median(axis=1)
    # écart du capteur le plus éloigné des deux autres, relatif (un capteur qui décroche ressort nettement)
    d["ecart"] = (d[["L1", "L2", "L3"]].sub(med, axis=0).abs().max(axis=1) / np.maximum(med, 2.0)).clip(0, 5)
    return {"points": d, "controles": controles(d, gnss, L, dt), "meta": meta, "source": source,
            "duree_lecture": round(time.time() - t0, 1)}


def controles(d: pd.DataFrame, gnss: pd.DataFrame, L: np.ndarray, dt: float) -> list[dict]:
    """Contrôles rapides de la tournée : ok / attention, avec une phrase pour l'opérateur."""
    c = []
    t = gnss["t"].to_numpy()
    trous = np.diff(t)
    gros = trous[trous > 2.0]
    c.append(dict(titre="Couverture GPS", ok=len(gros) == 0,
                  detail=f"{len(gnss)} positions · {len(gros)} coupure(s) de plus de 2 s"
                         + (f" (la plus longue : {gros.max():.0f} s)" if len(gros) else "")))
    km = np.nansum(np.hypot(np.diff(gnss["x"]), np.diff(gnss["y"]))) / 1000
    c.append(dict(titre="Trajet", ok=True,
                  detail=f"{km:.1f} km en {(t[-1] - t[0]) / 3600:.1f} h · vitesse médiane "
                         f"{3.6 * np.median(gnss['v'][gnss['v'] > 0.5]):.0f} km/h"))
    hd = float(np.nanmedian(gnss["hdop"])) if "hdop" in gnss else np.nan
    c.append(dict(titre="Précision GPS", ok=not (hd > 2.5),
                  detail=f"HDOP médian {hd:.1f} · {np.nanmedian(gnss['nsat']):.0f} satellites"))
    for k, nom in enumerate(CAPTEURS.values()):
        v = L[k]
        neg = float((v < -0.5).mean() * 100)
        # capteur figé : la valeur ne bouge plus pendant au moins 30 s
        bloc = int(30 / dt)
        fige = bool(len(v) > bloc and (pd.Series(v).rolling(bloc).std() < 1e-6).any())
        c.append(dict(titre=f"Capteur {nom}", ok=neg < 1 and not fige,
                      detail=f"médiane {np.median(v):.1f} lx · max {v.max():.0f} lx · {neg:.1f} % de valeurs négatives"
                             + (" · valeur figée pendant 30 s ou plus" if fige else "")))
    part = float((d["ecart"] > 0.5).mean() * 100)
    c.append(dict(titre="Accord entre les 3 capteurs", ok=part < 5,
                  detail=f"{part:.1f} % des positions avec un capteur à plus de 50 % des deux autres"))
    return c


def trace(brut: dict) -> dict:
    """Trace compacte (en colonnes) envoyée au navigateur : positions, cap, heure locale, vitesse et les valeurs
    brutes des 3 capteurs, avec leur position latérale lue dans le TDMS (lux_position, y > 0 = à gauche)."""
    d = brut["points"]
    t = d["t"].to_numpy()
    utc = datetime.fromtimestamp(float(t[0]), timezone.utc) if len(t) else datetime.now(timezone.utc)
    decal = utc.astimezone(HEURE_LOCALE).utcoffset().total_seconds()
    capteurs = []
    for k, (cle, m) in enumerate(zip(("L1", "L2", "L3"), brut["meta"])):
        y = float(m["position"][1])
        cote = "droite" if y < -0.05 else "gauche" if y > 0.05 else "centre"
        capteurs.append({"nom": f"{cle} ({cote})", "y": y})
    return {"lat": d["lat"].round(6).tolist(), "lon": d["lon"].round(6).tolist(),
            "cap": d["cap"].round(0).tolist(), "h": ((t + decal) % 86400).round(1).tolist(),
            "v": (3.6 * d["v"]).round(0).tolist(), "capteurs": capteurs,
            "lux": [d[c].round(2).tolist() for c in ("L1", "L2", "L3")]}


# ------------------------------------------------------------------ ANNOTATIONS
def _fichier_notes(nom_tdms: str) -> Path:
    tdms = chemin_tdms(nom_tdms)
    return (tdms.parent if tdms else RECUS) / f"{Path(nom_tdms).stem}.notes.json"


def lire_notes(nom_tdms: str | None) -> dict:
    vide = {"points": [], "notes": ""}
    if not nom_tdms:
        return vide
    f = _fichier_notes(nom_tdms)
    if not f.exists():
        return vide
    try:
        return {**vide, **json.loads(f.read_text(encoding="utf-8"))}
    except ValueError:
        return vide


def ecrire_notes(nom_tdms: str, notes: dict) -> None:
    f = _fichier_notes(nom_tdms)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")


def notes_du_rapport(fichier_tdms: str | None) -> dict:
    """Notes de l'opérateur pour un rapport (resume.json -> meta.fichier_tdms)."""
    return lire_notes(Path(fichier_tdms).name) if fichier_tdms else {"points": [], "notes": ""}
