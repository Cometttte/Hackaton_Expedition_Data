/* Fonctions de style de la carte Leaflet (dash-leaflet), exécutées dans le navigateur.
   Couleurs : palette froid -> chaud de carte.html (getys_rapport.py), 0 à 50 lx.
   Les seuils SDAL et le mode de couleur arrivent par le « hideout » du calque : les changer recolore
   la carte sans renvoyer de données. */
window.getys = Object.assign({}, window.getys, (function () {
    const STOPS = [[0, [15, 59, 76]], [5, [30, 95, 105]], [10, [52, 135, 120]], [15, [140, 200, 160]],
                   [22, [232, 236, 160]], [30, [247, 214, 120]], [40, [246, 170, 95]], [50, [238, 120, 70]]];
    function rgb(v) {
        v = Math.max(0, Math.min(50, v == null ? 0 : v));
        for (let i = 1; i < STOPS.length; i++) {
            if (v <= STOPS[i][0]) {
                const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i], t = (v - a) / (b - a);
                const c = ca.map((x, k) => Math.round(x + (cb[k] - x) * t));
                return `rgb(${c[0]},${c[1]},${c[2]})`;
            }
        }
        const c = STOPS[STOPS.length - 1][1];
        return `rgb(${c[0]},${c[1]},${c[2]})`;
    }
    // classes d'éclairement des tronçons : couleur au milieu de chaque classe (comme la légende)
    const PALIERS = [0, 2, 5, 7.5, 10, 15, 20, 30, Infinity];
    const MILIEUX = [1, 3.5, 6.25, 8.75, 12.5, 17.5, 25, 40];
    function palier(v) {
        for (let i = 0; i < MILIEUX.length; i++) if (v < PALIERS[i + 1]) return rgb(MILIEUX[i]);
        return rgb(40);
    }
    // conformité SDAL (mêmes valeurs que rapport_affaires.py) : sous le seuil en bleu (bleu clair : un seul sens
    // mesuré), conforme en vert, et en rouge les tronçons à plus de 40 lx
    const SOUS = '#2F6FDB', SOUS_1S = '#8DB7F2', CONFORME = '#2E9E5B', ROUGE_FORT = '#E03C31', LUX_ROUGE = 40;
    // noms de rue comparés sans accents ni majuscules (même règle que sdal_fichier.normaliser)
    const normaliser = s => String(s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ').trim();
    const TYPES = {sodium: '#ff8a2a', blanc_2700: '#ffd23f', led_3000: '#3b9bff', blanc_neutre: '#2ad4b0',
                   blanc_froid: '#b18cff', inconnu: '#a0a8b5'};
    const fmt = (v, d = 1) => v == null ? '–' : Number(v).toLocaleString('fr-FR',
        {minimumFractionDigits: d, maximumFractionDigits: d});

    function etat(p, h) {
        const s = ((h || {}).sdal || {})[p.c], rue = ((h || {}).rues || {})[normaliser(p.n)];
        const seuil = rue != null ? rue : (s == null ? p.sd : s);      // seuil de la rue (fichier SDAL) d'abord
        return {seuil, conforme: p.E >= seuil, incomplet: p.ns < 2 && !p.su};
    }
    // --- taille des points chauds / froids selon le zoom
    const POINTS = [];
    let zoomSuivi = false;
    function rayonPoints() {
        const m = window.getysCarte;
        if (!m) return 3;
        const mpp = 40075016.686 * Math.cos(m.getCenter().lat * Math.PI / 180) / Math.pow(2, m.getZoom() + 8);
        return Math.max(2, Math.min(9, 9 / mpp));      // 9 m au sol, entre 2 et 9 px de rayon
    }
    function suivreZoom() {
        const m = window.getysCarte;
        if (zoomSuivi || !m) return;
        zoomSuivi = true;
        m.on('zoomend', () => {
            const r = rayonPoints();
            let j = 0;
            for (const c of POINTS) { if (c._map) { c.setRadius(r); POINTS[j++] = c; } }   // oublie les retirés
            POINTS.length = j;
        });
    }
    // UN SEUL rendu canvas pour tous les calques (tronçons, hexagones, points, lampadaires, sélection), avec une
    // tolérance de clic de 6 px. Avec Leaflet 1.9, deux canvas superposés se bloquent : seul celui du dessus
    // reçoit les clics (corrigé dans Leaflet 2.0). Il devient le rendu par défaut de la carte dès sa création.
    const renduTolerant = () => L.canvas({tolerance: 6, padding: 0.5});    // un rendu par carte
    // carte Leaflet accessible (window.getysCarte) : utile pour le débogage et les tests
    (function attendreLeaflet() {
        if (window.L && L.Map) {
            L.Map.addInitHook(function () {
                this.options.renderer = renduTolerant();
                // vue Opérateur : carte à part (window.getysCarteOperateur), sans fond vectoriel
                if (this._container && this._container.id === 'op-carte') window.getysCarteOperateur = this;
                else window.getysCarte = this;
            });
        } else { setTimeout(attendreLeaflet, 20); }
    })();
    // mémorise le point cliqué : il part avec clickData (Street View à l'endroit du clic)
    function retenirClic(feature, layer) {
        layer.on('click', e => { feature.properties._lat = e.latlng.lat; feature.properties._lon = e.latlng.lng; });
    }

    return {
        rgb: rgb,
        styleTroncon: function (feature, context) {
            const h = context.hideout || {}, p = feature.properties;
            let color = palier(p.E);
            if (h.couleur === 'conformite') {
                const e = etat(p, h);
                color = e.conforme ? (p.E > LUX_ROUGE ? ROUGE_FORT : CONFORME) : (e.incomplet ? SOUS_1S : SOUS);
            }
            return {color: color, weight: 4, opacity: 0.95, lineCap: 'round'};
        },
        infoTroncon: function (feature, layer, context) {
            retenirClic(feature, layer);
            layer.bindTooltip(() => {
                const p = feature.properties, e = etat(p, context.hideout);
                const sens = p.su ? `sens unique : ${fmt(p.E)} lx (${p.np} passage(s))`
                    : `sens 1 : ${fmt(p.s1)} lx (${p.n1} p.) · sens 2 : ${fmt(p.s2)} lx (${p.n2} p.)`;
                return `<b>${p.n}</b><br>${p.lib} — seuil ${fmt(e.seuil)} lx<br>Moyenne des deux sens : ` +
                    `<b>${fmt(p.E)} lx</b><br>${sens}<br>${!e.conforme ? '🔵 sous le seuil' : p.E > LUX_ROUGE ? `🔴 plus de ${LUX_ROUGE} lx` : '🟢 conforme'}`;
            }, {sticky: true});
        },
        styleHex: function (feature) {
            // smoothFactor 0 : sans ça, Leaflet simplifie les hexagones de 2 m (2-3 px au zoom 16-17) jusqu'à les effacer
            return {fillColor: rgb(feature.properties.E), fillOpacity: 0.95,
                    color: 'rgba(110,115,125,0.7)', weight: 0.4, smoothFactor: 0};
        },
        infoHex: function (feature, layer) { retenirClic(feature, layer); },
        // points chauds / froids, comme carte.html : rayon de 9 m converti en pixels, borné, recalculé à chaque zoom
        // (un cercle en mètres grossit sans fin quand on zoome)
        pointChaud: function (feature, latlng) {
            const c = L.circleMarker(latlng, {radius: rayonPoints(), stroke: false, interactive: true,
                                              fillColor: rgb(feature.properties.E), fillOpacity: 0.6});
            const p = feature.properties;
            c.bindTooltip(() => `<b>${fmt(p.E)} lx</b><br>sens ${p.s} · ${p.h}`, {direction: 'top'});
            POINTS.push(c);
            suivreZoom();
            return c;
        },
        lampe: function (feature, latlng) {
            const p = feature.properties;
            return L.circleMarker(latlng, {radius: 5.5, weight: 1.2, color: '#111',
                                           fillColor: TYPES[p.t] || '#ffffff', fillOpacity: 1});
        },
        infoLampe: function (feature, layer) {
            const p = feature.properties;
            layer.bindTooltip(() => `💡 <b>${p.lib}</b><br>Puissance estimée : ${fmt(p.w, 0)} W ` +
                `(${fmt(p.wmin, 0)}–${fmt(p.wmax, 0)})<br>Côté : ${p.cote} · pic ${fmt(p.pic, 0)} lx · ` +
                `${p.np} passage(s)`);
        },
    };
})());

/* Vue Opérateur : les 3 luxmètres bruts, côte à côte le long de la trace. ≈ 180 000 points : dessinés sur un canevas
   unique (bien plus rapide que des marqueurs Leaflet), redessiné après chaque déplacement ou zoom. */
window.getys = Object.assign({}, window.getys, (function () {
    const COULEURS = Array.from({length: 51}, (_, v) => window.getys.rgb(v));
    let calque = null, demande = null;

    function creer(d) {
        const Calque = L.Layer.extend({
            onAdd: function (m) {
                const volet = m.getPane('traceBrute') || m.createPane('traceBrute');
                volet.style.zIndex = 450;
                volet.style.pointerEvents = 'none';               // les clics vont à la carte (points annotés)
                this._canevas = L.DomUtil.create('canvas', 'leaflet-zoom-hide', volet);
                this._bulle = L.tooltip({direction: 'top', offset: [0, -6]});
                this._ecrans = [];
                m.on('moveend zoomend resize viewreset', this._planifier, this);
                m.on('mousemove', this._survol, this);
                m.on('mouseout', this._cacher, this);
                this._planifier();
            },
            // plusieurs événements arrivent ensemble (zoom, déplacement, redimensionnement) : un seul dessin
            _planifier: function () {
                clearTimeout(this._attente);
                this._attente = setTimeout(() => { if (this._map) this._dessiner(); }, 60);
            },
            onRemove: function (m) {
                m.off('moveend zoomend resize viewreset', this._planifier, this);
                clearTimeout(this._attente);
                m.off('mousemove', this._survol, this);
                m.off('mouseout', this._cacher, this);
                this._cacher();
                L.DomUtil.remove(this._canevas);
            },
            nombre: function () { return d.lat.length * 3; },
            dessines: function () { return this._ecrans.length; },
            _dessiner: function () {
                const m = this._map, taille = m.getSize(), c = this._canevas, ratio = window.devicePixelRatio || 1;
                if (!taille.x || !taille.y) return;
                c.width = taille.x * ratio; c.height = taille.y * ratio;
                c.style.width = taille.x + 'px'; c.style.height = taille.y + 'px';
                L.DomUtil.setPosition(c, m.containerPointToLayerPoint([0, 0]));
                const ctx = c.getContext('2d');
                ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
                const z = m.getZoom();
                const mpp = 40075016.686 * Math.cos(m.getCenter().lat * Math.PI / 180) / Math.pow(2, z + 8);
                // pixels par mètre en travers de la route : échelle réelle, avec un minimum pour garder 3 rangées lisibles
                const ecart = Math.max(...d.capteurs.map(k => Math.abs(k.y))) || 1;
                const parMetre = Math.max(1 / mpp, (z >= 17 ? 6 : z >= 15.5 ? 4.5 : 3.2) / ecart);
                const rayon = z >= 19 ? 4 : z >= 17 ? 2.8 : z >= 15.5 ? 2 : 1.4;
                const lots = COULEURS.map(() => []);
                this._ecrans = [];
                for (let i = 0; i < d.lat.length; i++) {
                    const p = m.latLngToContainerPoint([d.lat[i], d.lon[i]]);
                    if (p.x < -20 || p.y < -20 || p.x > taille.x + 20 || p.y > taille.y + 20) continue;
                    const a = d.cap[i] * Math.PI / 180, gx = -Math.cos(a), gy = -Math.sin(a);   // « à gauche » à l'écran
                    for (let k = 0; k < 3; k++) {
                        const v = d.lux[k][i];
                        if (v == null) continue;
                        const x = p.x + gx * d.capteurs[k].y * parMetre, y = p.y + gy * d.capteurs[k].y * parMetre;
                        lots[Math.max(0, Math.min(50, Math.round(v)))].push(x, y);
                        this._ecrans.push([x, y, i, k]);
                    }
                }
                lots.forEach((pts, v) => {
                    if (!pts.length) return;
                    ctx.fillStyle = COULEURS[v];
                    if (rayon <= 2) {                    // points minuscules : des carrés, bien plus rapides, même aspect
                        for (let j = 0; j < pts.length; j += 2)
                            ctx.fillRect(pts[j] - rayon, pts[j + 1] - rayon, 2 * rayon, 2 * rayon);
                        return;
                    }
                    ctx.beginPath();
                    for (let j = 0; j < pts.length; j += 2) {
                        ctx.moveTo(pts[j] + rayon, pts[j + 1]);
                        ctx.arc(pts[j], pts[j + 1], rayon, 0, 2 * Math.PI);
                    }
                    ctx.fill();
                });
            },
            _survol: function (e) {
                let proche = null, dmin = 64;                       // à moins de 8 px
                for (const q of this._ecrans) {
                    const dist = (q[0] - e.containerPoint.x) ** 2 + (q[1] - e.containerPoint.y) ** 2;
                    if (dist < dmin) { dmin = dist; proche = q; }
                }
                if (!proche) return this._cacher();
                const i = proche[2], s = d.h[i];
                const deux = n => String(Math.floor(n)).padStart(2, '0');
                const heure = `${deux(s / 3600)}:${deux(s % 3600 / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;
                const lignes = [2, 1, 0].map(k => {
                    const v = d.lux[k][i], t = `${d.capteurs[k].nom} : ${v.toLocaleString('fr-FR')} lx`;
                    return k === proche[3] ? `<b>${t}</b>` : t;
                });
                this._bulle.setLatLng([d.lat[i], d.lon[i]]).setContent(`<b>${heure}</b> · ${d.v[i]} km/h<br>` + lignes.join('<br>'));
                this._bulle.addTo(this._map);
            },
            _cacher: function () { if (this._bulle) this._bulle.remove(); },
        });
        return new Calque();
    }

    return {
        traceBrute: async function (url) {
            if (url === demande) return;                            // déjà affichée ou en cours de chargement
            demande = url;
            const m = await new Promise(ok => { (function attendre() {
                window.getysCarteOperateur ? ok(window.getysCarteOperateur) : setTimeout(attendre, 50); })(); });
            if (calque) { m.removeLayer(calque); calque = null; window.getysTrace = null; }
            if (!url) return;
            const d = await (await fetch(url)).json();
            if (demande !== url) return;                            // un autre fichier a été choisi entre-temps
            calque = creer(d).addTo(m);
            window.getysTrace = calque;
        },
    };
})());

/* Fond de plan vectoriel (OpenFreeMap, sans clé) affiché DANS Leaflet par l'extension maplibre-gl-leaflet :
   Leaflet reste le moteur de la carte (calques, clics, zoom) ; MapLibre ne dessine que le fond, net à tous
   les zooms et en haute résolution. Satellite : tuiles raster Esri (zoom 19 max). */
window.getys = Object.assign({}, window.getys, (function () {
    const STYLES = {clair: 'https://tiles.openfreemap.org/styles/positron',
                    sombre: 'https://tiles.openfreemap.org/styles/dark'};
    const ATTRIBUTION = '<a href="https://openfreemap.org" target="_blank">OpenFreeMap</a> © ' +
        '<a href="https://www.openmaptiles.org/" target="_blank">OpenMapTiles</a> © ' +
        '<a href="https://www.openstreetmap.org/copyright" target="_blank">contributeurs OpenStreetMap</a>';
    const LIBS = ['https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js',
                  'https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.1.4/leaflet-maplibre-gl.js'];
    let chargement = null;

    function charger(url) {
        return new Promise((ok, ko) => {
            const s = document.createElement('script');
            s.src = url; s.onload = ok; s.onerror = ko; document.head.appendChild(s);
        });
    }
    // MapLibre puis l'extension, une seule fois, quand Leaflet (window.L) est prêt
    function bibliotheques() {
        if (!chargement) {
            chargement = new Promise(ok => {
                (function attendre() { (window.L && L.Map) ? ok() : setTimeout(attendre, 30); })();
            }).then(() => charger(LIBS[0])).then(() => charger(LIBS[1]));
        }
        return chargement;
    }
    function carte(nom) {
        return new Promise(ok => { (function attendre() {
            window[nom] ? ok(window[nom]) : setTimeout(attendre, 50); })(); });
    }
    // une carte d'une page masquée n'a pas de taille : MapLibre n'y est créé qu'une fois la carte visible
    function visible(m) {
        return new Promise(ok => { (function attendre() {
            m.getContainer().offsetWidth ? ok() : setTimeout(attendre, 150); })(); });
    }
    async function appliquer(m, fond) {
        const style = STYLES[fond] || null;
        m.__fondVoulu = fond;
        // satellite : zoom natif max des images ; vectoriel : net jusqu'à la rue
        m.setMaxZoom(style ? 21 : 19);
        if (!style && m.getZoom() > 19) m.setZoom(19);
        if (style === (m.__fondStyle || null)) return;
        if (style) { await visible(m); await bibliotheques(); }
        if (m.__fondVoulu !== fond || style === (m.__fondStyle || null)) return;       // changé entre-temps
        if (m.__fondCalque) { m.removeLayer(m.__fondCalque); m.__fondCalque = null; }
        m.__fondStyle = style;
        if (style) {
            m.__fondCalque = L.maplibreGL({style: style, attribution: ATTRIBUTION, interactive: false});
            m.__fondCalque.addTo(m);
        }
    }

    return {
        // le même fond pour la carte de l'analyste et celle de l'opérateur
        appliquerFond: function (fond) {
            for (const nom of ['getysCarte', 'getysCarteOperateur']) carte(nom).then(m => appliquer(m, fond));
            return fond;
        },
    };
})());
// Dash cherche les fonctions client dans window.dash_clientside.<namespace>
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    getys: {appliquerFond: (fond) => window.getys.appliquerFond(fond)},
});
