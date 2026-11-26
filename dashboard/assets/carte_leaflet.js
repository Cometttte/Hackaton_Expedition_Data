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
    const VERT = '#00B894', ROUGE = '#E75113', ORANGE = '#F7A600';   // charte Bouygues E&S
    const TYPES = {sodium: '#ff8a2a', blanc_2700: '#ffd23f', led_3000: '#3b9bff', blanc_neutre: '#2ad4b0',
                   blanc_froid: '#b18cff', inconnu: '#a0a8b5'};
    const fmt = (v, d = 1) => v == null ? '–' : Number(v).toLocaleString('fr-FR',
        {minimumFractionDigits: d, maximumFractionDigits: d});

    function etat(p, h) {
        const s = ((h || {}).sdal || {})[p.c];
        const seuil = s == null ? p.sd : s;
        return {seuil, conforme: p.E >= seuil, incomplet: p.ns < 2 && !p.su};
    }
    // rendu canvas avec une tolérance de clic de 6 px : inutile de viser pile le trait de 4 px
    let rendu = null;
    const renduTolerant = () => rendu || (rendu = L.canvas({tolerance: 6, padding: 0.5}));
    // carte Leaflet accessible (window.getysCarte) : utile pour le débogage et les tests
    (function attendreLeaflet() {
        if (window.L && L.Map) { L.Map.addInitHook(function () { window.getysCarte = this; }); }
        else { setTimeout(attendreLeaflet, 20); }
    })();
    // mémorise le point cliqué : il part avec clickData (Street View à l'endroit du clic)
    function retenirClic(feature, layer) {
        layer.on('click', e => { feature.properties._lat = e.latlng.lat; feature.properties._lon = e.latlng.lng; });
    }

    return {
        styleTroncon: function (feature, context) {
            const h = context.hideout || {}, p = feature.properties;
            let color = palier(p.E);
            if (h.couleur === 'conformite') {
                const e = etat(p, h);
                color = e.conforme ? VERT : (e.incomplet ? ORANGE : ROUGE);
            }
            return {color: color, weight: 4, opacity: 0.95, lineCap: 'round', renderer: renduTolerant()};
        },
        infoTroncon: function (feature, layer, context) {
            retenirClic(feature, layer);
            layer.bindTooltip(() => {
                const p = feature.properties, e = etat(p, context.hideout);
                const sens = p.su ? `sens unique : ${fmt(p.E)} lx (${p.np} passage(s))`
                    : `sens 1 : ${fmt(p.s1)} lx (${p.n1} p.) · sens 2 : ${fmt(p.s2)} lx (${p.n2} p.)`;
                return `<b>${p.n}</b><br>${p.lib} — seuil ${fmt(e.seuil)} lx<br>Moyenne des deux sens : ` +
                    `<b>${fmt(p.E)} lx</b><br>${sens}<br>${e.conforme ? '✅ conforme' : '❌ non conforme'}`;
            }, {sticky: true});
        },
        styleHex: function (feature) {
            // smoothFactor 0 : sans ça, Leaflet simplifie les hexagones de 2 m (2-3 px au zoom 16-17) jusqu'à les effacer
            return {fillColor: rgb(feature.properties.E), fillOpacity: 0.95,
                    color: 'rgba(110,115,125,0.7)', weight: 0.4, smoothFactor: 0};
        },
        infoHex: function (feature, layer) { retenirClic(feature, layer); },
        // points chauds / froids : cercle de 9 m de rayon (en mètres, donc constant au sol quel que soit le zoom)
        pointChaud: function (feature, latlng) {
            return L.circle(latlng, {radius: 9, stroke: false, fillColor: rgb(feature.properties.E),
                                     fillOpacity: 0.55, interactive: true});
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
    let chargement = null, calque = null, styleActif = null;

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
    function carte() {
        return new Promise(ok => { (function attendre() {
            window.getysCarte ? ok(window.getysCarte) : setTimeout(attendre, 50); })(); });
    }

    return {
        appliquerFond: async function (fond) {
            const m = await carte();
            const style = STYLES[fond] || null;
            if (style === styleActif) return fond;
            if (calque) { m.removeLayer(calque); calque = null; }
            styleActif = style;
            if (style) {
                await bibliotheques();
                if (styleActif !== style) return fond;            // changé entre-temps
                calque = L.maplibreGL({style: style, attribution: ATTRIBUTION, interactive: false});
                calque.addTo(m);
                m.setMaxZoom(21);                                 // vectoriel : net jusqu'à la rue
            } else {
                m.setMaxZoom(19);                                 // satellite : zoom natif max des images
                if (m.getZoom() > 19) m.setZoom(19);
            }
            return fond;
        },
    };
})());
// Dash cherche les fonctions client dans window.dash_clientside.<namespace>
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    getys: {appliquerFond: (fond) => window.getys.appliquerFond(fond)},
});
