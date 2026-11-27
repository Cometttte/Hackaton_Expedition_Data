"""Rapport du chargé d'affaires : page HTML autonome (imprimable en PDF depuis le navigateur), composée des
sections cochées dans la vue « Chargé d'affaires ». Une première sélection est proposée (DEFAUT)."""
from __future__ import annotations

import html as h
import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.io as pio

# Couleurs de conformité (mêmes que la carte et assets/carte_leaflet.js) : sous le seuil SDAL en bleu, conforme en
# vert, et en rouge les tronçons à plus de LUX_ROUGE lx
SOUS, SOUS_1S, CONFORME, ROUGE_FORT = "#2F6FDB", "#8DB7F2", "#2E9E5B", "#E03C31"
LUX_ROUGE = 40
VERT_FONCE, VERT = "#1F6E43", "#3E9C61"

SECTIONS = [
    ("synthese", "Synthèse et indicateurs clés", True),
    ("carte", "Carte de conformité des tronçons", True),
    ("deficits", "Rues les plus en déficit", True),
    ("types", "Conformité par type de voie", True),
    ("non_conformes", "Liste des tronçons sous le seuil", False),
    ("lampadaires", "Parc de lampadaires estimé", True),
    ("chronologie", "Chronologie de la tournée", False),
    ("qualite", "Qualité des données", False),
    ("bruit", "Bruit et valeurs aberrantes", False),
    ("operateur", "Points et notes de l'opérateur", True),
    ("methode", "Méthode de mesure", False),
]
DEFAUT = [s for s, _, coche in SECTIONS if coche]


def fmt(v, suffixe="", dec=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",") + suffixe


def e(t):
    return h.escape(str(t if t is not None else ""))


# ------------------------------------------------------------------ BLOCS
def deficits(tr: pd.DataFrame, n=10) -> pd.DataFrame:
    """Rues classées par déficit cumulé : Σ longueur × (1 − E/seuil) sur leurs tronçons sous le seuil."""
    t = tr.assign(manque=(1 - tr["E_moy"] / tr["seuil"]).clip(lower=0), lg_nc=tr["longueur_m"].where(~tr["conforme"], 0),
                  e_lg=tr["E_moy"] * tr["longueur_m"], s_lg=tr["seuil"] * tr["longueur_m"])
    t["score"] = t["manque"] * t["longueur_m"]
    g = t.groupby("nom").agg(score=("score", "sum"), lg=("longueur_m", "sum"), lg_nc=("lg_nc", "sum"),
                             e_lg=("e_lg", "sum"), s_lg=("s_lg", "sum"), n=("id", "size"),
                             n_nc=("conforme", lambda c: int((~c).sum())))
    g["E"], g["S"] = g["e_lg"] / g["lg"], g["s_lg"] / g["lg"]
    return g[g["lg_nc"] > 0].sort_values("score", ascending=False).head(n)


def carte_svg(tr: pd.DataFrame, geoms: dict, points=(), l=900, hmax=560) -> str:
    """Tronçons colorés par conformité (bleu : sous le seuil, rouge : au-dessus), projection locale."""
    lon = np.concatenate([geoms[i][0] for i in tr["id"]])
    lat = np.concatenate([geoms[i][1] for i in tr["id"]])
    k = math.cos(math.radians(float(lat.mean())))
    x0, x1, y0, y1 = lon.min() * k, lon.max() * k, lat.min(), lat.max()
    ech = min((l - 20) / (x1 - x0), (hmax - 20) / (y1 - y0))
    haut = (y1 - y0) * ech + 20
    xy = lambda lo, la: (10 + (lo * k - x0) * ech, 10 + (y1 - la) * ech)  # noqa: E731
    traits = []
    for r in tr.itertuples():
        glo, gla = geoms[r.id]
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in (xy(lo, la) for lo, la in zip(glo, gla)))
        if r.conforme:
            c = ROUGE_FORT if r.E_moy > LUX_ROUGE else CONFORME
        else:
            c = SOUS_1S if r.a_completer else SOUS
        traits.append(f'<polyline points="{pts}" stroke="{c}" stroke-width="2.4" fill="none" stroke-linecap="round">'
                      f'<title>{e(r.nom)} — {fmt(r.E_moy, " lx")} (seuil {fmt(r.seuil, " lx")})</title></polyline>')
    for i, p in enumerate(points, 1):
        a, b = xy(p["lon"], p["lat"])
        traits.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" fill="#D63AF9" stroke="#fff" stroke-width="1.5"/>'
                      f'<text x="{a:.1f}" y="{b + 3.5:.1f}" font-size="9" text-anchor="middle" font-weight="700" fill="#fff">{i}</text>')
    return f'<svg viewBox="0 0 {l} {haut:.0f}" width="100%" role="img" aria-label="Carte de conformité">{"".join(traits)}</svg>'


def _table(entetes, lignes, num=()):
    th = "".join(f'<th class="{"n" if i in num else ""}">{e(c)}</th>' for i, c in enumerate(entetes))
    tb = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{c}</td>' for i, c in enumerate(l)) + "</tr>"
                 for l in lignes)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table>"


def _figure(fig, hauteur=320):
    fig.update_layout(height=hauteur, template="plotly_white", paper_bgcolor="white")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


# ------------------------------------------------------------------ RAPPORT
def construire(sections, infos, data, tr, classes, figures, notes) -> str:
    """sections : ids cochés ; infos : titre, destinataire, auteur, commentaire ; data : charger() du dashboard ;
    tr : tronçons avec seuils (avec_seuils) ; classes : troncons.CLASSES ; figures : {"lampes", "chrono"} (Plotly) ;
    notes : notes de l'opérateur {"points": [...], "notes": str}."""
    meta, k = data["resume"].get("meta", {}), data["resume"].get("kpi", {})
    q = data["resume"].get("qualite", {})
    s = set(sections)
    corps = []
    titre = infos.get("titre") or f"Diagnostic de l'éclairage public — {meta.get('ville', data['nom'])}"

    if "synthese" in s and tr is not None and len(tr):
        lg = tr["longueur_m"]
        part = 100 * lg[tr["conforme"]].sum() / lg.sum()
        e_moy = (tr["E_moy"] * lg).sum() / lg.sum()
        tuiles = [("Linéaire au-dessus du seuil", fmt(part, " %", 0), f"{fmt(lg[tr['conforme']].sum() / 1000, ' km')} sur {fmt(lg.sum() / 1000, ' km')}"),
                  ("Tronçons au-dessus du seuil", f"{int(tr['conforme'].sum())} / {len(tr)}", f"{int((~tr['conforme']).sum())} sous le seuil SDAL"),
                  ("Éclairement moyen", fmt(e_moy, " lx"), "moyenne des tronçons pondérée par la longueur"),
                  ("Distance mesurée", fmt(k.get("dist_km"), " km"), f"en {fmt(k.get('duree_h'), ' h')}"),
                  ("Lampadaires estimés", fmt(k.get("n_lampadaires"), dec=0), f"{fmt(k.get('puissance_tot_kw'), ' kW')} installés"),
                  ("Tronçons à repasser", f"{int(tr['a_completer'].sum())}", "double sens mesuré dans un seul sens")]
        corps.append("<h2>Synthèse</h2><div class='tuiles'>" + "".join(
            f"<div class='tuile'><div class='t-titre'>{e(a)}</div><div class='t-val'>{b}</div><div class='t-det'>{e(c)}</div></div>"
            for a, b, c in tuiles) + "</div>")

    if "carte" in s and tr is not None and len(tr):
        pts = notes.get("points", []) if "operateur" in s else []
        corps.append(f"<h2>Carte de conformité des tronçons</h2>{carte_svg(tr, data['geoms'], pts)}"
                     f"<p class='leg'><i style='background:{ROUGE_FORT}'></i>plus de {LUX_ROUGE} lx "
                     f"<i style='background:{CONFORME}'></i>conforme jusqu'à {LUX_ROUGE} lx "
                     f"<i style='background:{SOUS}'></i>sous le seuil <i style='background:{SOUS_1S}'></i>sous le seuil, "
                     f"un seul sens mesuré" + (" <i class='rond'></i>point de l'opérateur" if pts else "") + "</p>")

    if "deficits" in s and tr is not None and len(tr):
        d = deficits(tr)
        lignes = [[f"{i}", e(n), fmt(r.lg_nc, " m", 0) + f" / {fmt(r.lg, ' m', 0)}", f"{r.n_nc} / {r.n}",
                   fmt(r.E, " lx"), fmt(r.S, " lx")] for i, (n, r) in enumerate(d.iterrows(), 1)]
        corps.append("<h2>Rues les plus en déficit</h2><p class='note'>Classement par déficit cumulé "
                     "(longueur × écart au seuil).</p>" +
                     (_table(["#", "Rue", "Linéaire sous le seuil", "Tronçons", "E moyen", "Seuil"], lignes, num=(2, 3, 4, 5))
                      if lignes else "<p>Toutes les rues respectent leur seuil SDAL.</p>"))

    if "types" in s and tr is not None and len(tr):
        lignes = []
        for c, (lib, _, _) in classes.items():
            t = tr[tr["classe"] == c]
            if t.empty:
                continue
            lg = t["longueur_m"]
            lignes.append([e(lib), fmt(t["seuil"].median(), " lx"), fmt(lg.sum() / 1000, " km", 2),
                           fmt(100 * lg[t["conforme"]].sum() / lg.sum(), " %", 0),
                           fmt((t["E_moy"] * lg).sum() / lg.sum(), " lx")])
        corps.append("<h2>Conformité par type de voie</h2>" +
                     _table(["Type de voie", "Seuil", "Linéaire", "Au-dessus du seuil", "E moyen"], lignes, num=(1, 2, 3, 4)))

    if "non_conformes" in s and tr is not None and len(tr):
        nc = tr[~tr["conforme"]].sort_values("E_moy")
        lignes = [[e(r.nom), e(classes[r.classe][0]), fmt(r.longueur_m, " m", 0), fmt(r.E_moy, " lx"), fmt(r.seuil, " lx"),
                   "un seul sens" if r.a_completer else ""] for r in nc.itertuples()]
        corps.append(f"<h2>Tronçons sous le seuil ({len(nc)})</h2>" +
                     _table(["Voie", "Type", "Longueur", "E moyen", "Seuil", "Remarque"], lignes, num=(2, 3, 4)))

    if "lampadaires" in s and data["resume"].get("types"):
        ty = sorted(data["resume"]["types"], key=lambda t: -t["n"])
        lignes = [[e(t["libelle"].split(":")[0]), f"{t['n']}", fmt(t.get("p_med"), " W", 0), fmt(t.get("p_tot_kw"), " kW"),
                   fmt(t.get("c_tot_kw"), " kW")] for t in ty]
        corps.append("<h2>Parc de lampadaires estimé</h2>" +
                     _table(["Type de lampe", "Nombre", "Puissance médiane", "Puissance installée", "Consommation"], lignes,
                            num=(1, 2, 3, 4)) + (_figure(figures["lampes"]) if figures.get("lampes") else ""))

    if "chronologie" in s and figures.get("chrono"):
        corps.append("<h2>Chronologie de la tournée</h2>" + _figure(figures["chrono"], 280))

    if "qualite" in s:
        lignes = [["Source GNSS", e(q.get("gnss_source"))], ["HDOP médian / satellites", f"{fmt(q.get('hdop_med'))} / {fmt(q.get('nsat_med'), dec=0)}"],
                  ["Décalage d'horloge corrigé", fmt(q.get("decalage_horloge_s"), " s")],
                  ["Mesures négatives", fmt(q.get("part_negatifs"), " %", 2)],
                  ["Tronçons OSM mesurés", f"{len(tr) if tr is not None else 0}"]]
        corps.append("<h2>Qualité des données</h2>" + _table(["Contrôle", "Valeur"], lignes))

    if "bruit" in s:
        ab = data.get("aberrants")
        n_ab = 0 if ab is None else len(ab)
        corps.append(f"<h2>Bruit et valeurs aberrantes</h2><p>{fmt(k.get('n_aberrants'), dec=0)} valeurs aberrantes retirées "
                     f"({fmt(q.get('part_aberrants'), ' %', 2)} des mesures), remplacées par interpolation ; "
                     f"{fmt(q.get('part_pics'), ' %', 2)} de parasites sur un seul capteur (filtre de Hampel).</p>" +
                     (_table(["Heure", "Motif", "Pic", "Fond"], [[e(a.heure), e(a.motif), fmt(a.pic_lx, " lx", 0), fmt(a.fond_lx, " lx", 0)]
                                                               for a in ab.itertuples()], num=(2, 3)) if n_ab else ""))

    if "operateur" in s and (notes.get("points") or notes.get("notes")):
        bloc = "<h2>Points et notes de l'opérateur</h2>"
        if notes.get("notes"):
            bloc += f"<div class='citation'>{e(notes['notes']).replace(chr(10), '<br>')}</div>"
        if notes.get("points"):
            bloc += _table(["#", "Commentaire", "Position"], [[f"{i}", e(p.get("texte")), f"{p['lat']:.5f}, {p['lon']:.5f}"]
                                                            for i, p in enumerate(notes["points"], 1)])
        corps.append(bloc)

    if "methode" in s:
        corps.append("<h2>Méthode de mesure</h2><p>Mesures embarquées GETYS : 3 luxmètres sur remorque à 1,5 m du sol, GNSS, "
                     "chromamètre. Signal filtré (filtre de Hampel, médiane des 3 capteurs, pics aberrants retirés), "
                     "ré-échantillonné tous les 2 m, rattaché au filaire OpenStreetMap. Éclairement d'un tronçon : moyenne de "
                     "chaque sens de circulation, puis moyenne des deux sens ; comparé au seuil SDAL de son type de voie "
                     "(ou de la rue, si un SDAL par rue est fourni).</p>")

    if infos.get("commentaire"):
        corps.append(f"<h2>Commentaire</h2><div class='citation'>{e(infos['commentaire']).replace(chr(10), '<br>')}</div>")

    entete = [f"{e(meta.get('ville', ''))}", f"tournée du {e(meta.get('date', '–'))}",
              f"{e(meta.get('debut', ''))} → {e(meta.get('fin', ''))}" if meta.get("debut") else ""]
    return PAGE.format(titre=e(titre), sous_titre=" · ".join(x for x in entete if x),
                       destinataire=e(infos.get("destinataire") or ""), auteur=e(infos.get("auteur") or ""),
                       date=date.today().strftime("%d/%m/%Y"), corps="\n".join(corps) or "<p>Aucune section sélectionnée.</p>")


PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>{titre}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{{font-family:Roboto,Arial,sans-serif;color:#26332B;max-width:960px;margin:0 auto;padding:28px 32px 60px;background:#fff}}
 header{{border-bottom:4px solid #8DBF2E;padding-bottom:14px;margin-bottom:20px}}
 h1{{color:#1F6E43;font-size:1.6rem;margin:0 0 4px}} .sous{{color:#3E9C61;font-weight:500}}
 .meta{{display:flex;gap:24px;flex-wrap:wrap;color:#66756B;font-size:.85rem;margin-top:8px}}
 h2{{color:#1F6E43;font-size:1.15rem;border-left:4px solid #3E9C61;padding-left:10px;margin:28px 0 10px;break-after:avoid}}
 .tuiles{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
 .tuile{{border:1px solid #DCE7DF;border-top:3px solid #3E9C61;border-radius:6px;padding:10px 12px}}
 .t-titre{{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#66756B;font-weight:700}}
 .t-val{{font-size:1.5rem;font-weight:700;color:#1F6E43}} .t-det{{font-size:.78rem;color:#66756B}}
 table{{width:100%;border-collapse:collapse;font-size:.85rem;margin:6px 0}}
 th{{background:#1F6E43;color:#fff;text-align:left;padding:6px 8px}} td{{border-bottom:1px solid #E3ECE5;padding:5px 8px}}
 .n{{text-align:right;font-variant-numeric:tabular-nums}} tr{{break-inside:avoid}}
 .note{{color:#66756B;font-size:.85rem}} .leg{{font-size:.8rem;color:#66756B}}
 .leg i{{display:inline-block;width:18px;height:4px;border-radius:2px;margin:0 6px 2px 12px;vertical-align:middle}}
 .leg i.rond{{width:10px;height:10px;border-radius:50%;background:#D63AF9;border:1px solid #fff}}
 .citation{{background:#F6FAF7;border-left:3px solid #8DBF2E;padding:10px 14px;border-radius:4px}}
 svg[role=img]{{border:1px solid #DCE7DF;border-radius:6px;background:#FAFCFA}}  /* la carte seule : pas les calques Plotly */
 @media print{{body{{padding:0}} .tuiles{{grid-template-columns:repeat(3,1fr)}} h2{{page-break-after:avoid}}}}
</style></head><body>
<header><h1>{titre}</h1><div class="sous">{sous_titre}</div>
<div class="meta"><span>Destinataire : <b>{destinataire}</b></span><span>Établi par : <b>{auteur}</b></span><span>Le {date}</span></div></header>
{corps}
<p class="note" style="margin-top:40px">Rapport généré par le tableau de bord Clair-Obscur (mesures GETYS). Pour un PDF : Fichier → Imprimer → Enregistrer au format PDF.</p>
</body></html>"""
