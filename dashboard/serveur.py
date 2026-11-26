"""Mini serveur local : reçoit un .tdms, lance getys_rapport.py (racine du dépôt), renvoie le dossier produit."""
import os
import re
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# ------------------------------------------------------------------ CONFIG
RACINE = Path(__file__).resolve().parent.parent
SCRIPT = RACINE / "getys_rapport.py"                                   # analyse de la tournée
RAPPORTS = Path(os.environ.get("RAPPORTS_DIR", RACINE / "rapports"))   # même dossier que le dashboard
RECUS = Path(os.environ.get("TDMS_RECUS_DIR", RACINE / "tdms_recus"))  # fichiers reçus

app = Flask(__name__)


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if f is None or not f.filename.lower().endswith(".tdms"):
        return jsonify(erreur="fichier .tdms manquant"), 400

    RECUS.mkdir(parents=True, exist_ok=True)
    chemin = RECUS / Path(f.filename).name
    f.save(chemin)

    cmd = [sys.executable, str(SCRIPT), str(chemin), "-o", str(RAPPORTS)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # Windows : évite les erreurs d'accents/symboles
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                         cwd=RACINE)
    if res.returncode != 0:
        return jsonify(erreur="échec du script", detail=res.stderr[-2000:]), 500

    # getys_rapport.py affiche « ✓ Rapport : <dossier>/rapport.html »
    trouve = re.search(r"Rapport : (.+?)[/\\]rapport\.html", res.stdout)
    if trouve:
        return jsonify(dossier=Path(trouve.group(1).strip()).name)
    dossiers = [d for d in RAPPORTS.iterdir() if d.is_dir() and (d / "mesures_pas2m.csv").exists()]
    return jsonify(dossier=max(dossiers, key=lambda d: d.stat().st_mtime).name)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "5001")))
