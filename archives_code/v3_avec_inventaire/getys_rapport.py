# -*- coding: utf-8 -*-
"""
GETYS - génération automatique du rapport de fin de tournée.

Usage :
  python getys_rapport.py TOURNEE.tdms INVENTAIRE.xlsx [-o DOSSIER]
  python getys_rapport.py --surveiller DOSSIER_TDMS INVENTAIRE.xlsx [-o DOSSIER]
      -> surveille un dossier et produit un rapport pour chaque nouveau .tdms (fin de tournée)

Sorties (dans DOSSIER/<ville>_<date>/ ; un rapport existant est déplacé dans DOSSIER/archives/, jamais écrasé) :
  rapport.html, carte.html (fond satellite, calques), mesures_pas2m.csv, hexagones.geojson, supports.csv, rues.csv,
  luminaires_detectes.geojson, resume.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from getys_analyse import analyser, Params  # noqa: E402

HEAD = ('<!doctype html><html lang="fr"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>body{margin:0}[hidden]{display:none!important}</style></head><body>\n')
TAIL = '\n</body></html>\n'


def rendre_html(res: dict, autonome: bool = True) -> str:
    tpl = (HERE / "rapport_template.html").read_text(encoding="utf-8")
    data = json.dumps({k: v for k, v in res.items() if k != "carte"}, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    body = tpl.replace("/*__DATA__*/null", data, 1)
    titre = f"Tournée {res['meta']['ville']} {res['meta']['date']}"
    body = body.replace("<title>Rapport de tournée luxmétrique</title>", f"<title>{titre}</title>", 1)
    return HEAD + body + TAIL if autonome else body


def rendre_carte(res: dict) -> str:
    tpl = (HERE / "carte_template.html").read_text(encoding="utf-8")
    m = res["meta"]
    meta = dict(ville=m["ville"], date=m["date"], debut=m["debut"], fin=m["fin"], seuil=m["params"]["seuil_emoy_lx"],
                recal=[m["params"]["recalage_est_m"], m["params"]["recalage_nord_m"]],
                hcap=m["params"]["hauteur_capteurs_m"], calage=res.get("calage"),
                seuil_pos=m["params"].get("ecart_position_min_m"), recal_inv=res.get("recalage_inventaire"))
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"), default=str).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", dump(res["carte"]), 1).replace("/*__META__*/null", dump(meta), 1)


def traiter(tdms: Path, xlsx: Path, out_root: Path, P: Params) -> Path:
    t0 = time.perf_counter()
    tmp = out_root / f"_en_cours_{tdms.stem}"
    res = analyser(tdms, xlsx, tmp, P)
    dest = out_root / f"{res['meta']['ville']}_{res['meta']['date_iso']}_{tdms.stem}"
    if dest.exists():
        # on n'écrase jamais un rapport : l'ancien est archivé avec sa date de génération
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(dest.stat().st_mtime))
        archive = out_root / "archives" / f"{dest.name}_genere_{stamp}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        dest.rename(archive)
        print(f"  ancien rapport archivé : {archive}")
    tmp.rename(dest)
    (dest / "rapport.html").write_text(rendre_html(res), encoding="utf-8")
    (dest / "carte.html").write_text(rendre_carte(res), encoding="utf-8")
    print(f"✓ Rapport : {dest / 'rapport.html'}  ({time.perf_counter() - t0:.1f} s)")
    return dest


def surveiller(dossier: Path, xlsx: Path, out_root: Path, P: Params, periode: float = 30.0):
    """Fin de tournée = un .tdms apparaît dans le dossier et sa taille ne bouge plus pendant une période."""
    vus = {p.name for p in dossier.glob("*.tdms") if (out_root / ".traites").exists()
           and p.name in (out_root / ".traites").read_text().split("\n")}
    tailles = {}
    print(f"Surveillance de {dossier} (Ctrl+C pour arrêter)")
    while True:
        for p in sorted(dossier.glob("*.tdms")):
            if p.name in vus:
                continue
            s = p.stat().st_size
            if tailles.get(p.name) == s:
                try:
                    traiter(p, xlsx, out_root, P)
                except Exception as e:  # on continue la surveillance
                    print(f"✗ {p.name} : {e}")
                vus.add(p.name)
                with open(out_root / ".traites", "a") as f:
                    f.write(p.name + "\n")
            tailles[p.name] = s
        time.sleep(periode)


def main():
    ap = argparse.ArgumentParser(description="Rapport automatique de fin de tournée GETYS")
    ap.add_argument("tdms", help="fichier .tdms de la tournée, ou dossier avec --surveiller")
    ap.add_argument("xlsx", help="export Excel des ouvrages")
    ap.add_argument("-o", "--sortie", default="rapports", help="dossier de sortie (défaut : ./rapports)")
    ap.add_argument("--surveiller", action="store_true", help="surveiller un dossier de .tdms")
    for f in fields(Params):
        typ = (lambda v: str(v).lower() in ("1", "true", "oui", "vrai")) if isinstance(f.default, bool) else type(f.default)
        ap.add_argument("--" + f.name.replace("_", "-"), type=typ, default=f.default)
    a = ap.parse_args()
    P = Params(**{f.name: getattr(a, f.name) for f in fields(Params)})
    out = Path(a.sortie)
    out.mkdir(parents=True, exist_ok=True)
    if a.surveiller:
        surveiller(Path(a.tdms), Path(a.xlsx), out, P)
    else:
        traiter(Path(a.tdms), Path(a.xlsx), out, P)


if __name__ == "__main__":
    main()
