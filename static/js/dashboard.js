/**
 * ResQFlow ML — Dashboard JavaScript (Station-Based Dispatch)
 * Handles: Leaflet map, station loading, condition chips,
 *          SUMO auto-launch, Socket.IO updates, route computation.
 */

// ── Socket.IO ──────────────────────────────────────────────────────────────
const socket = io();

// ── Map Setup ──────────────────────────────────────────────────────────────
let map = null;
let networkInfo = null;

const layers = {
    dijkstra: null,
    ml:       null,
    vehicles: null,
    markers:  null,
    stations: null,
};

// ── State ──────────────────────────────────────────────────────────────────
let stationsData   = [];          // full station list from backend
let selectedStation = null;       // currently selected station object
let dstLatLng      = null;        // destination {lat, lng}
let dstMarker      = null;
let lastResult     = null;

// Condition state (mirrors chip selections)
const conditions = {
    weather:       "clear",
    time_of_day:   "auto",
    incident_type: "medical",
    day_type:      "weekday",
    road_hazard:   "none",
};

// ── Init ───────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
    await initMap();
    await loadStations();
    await loadStatus();
    bindSocketEvents();
    startStatusPoller();
});

// ── Map ────────────────────────────────────────────────────────────────────
async function initMap() {
    try {
        const res = await fetch('/api/network/info');
        networkInfo = await res.json();
    } catch (e) {
        networkInfo = { center: { lat: 12.3051, lon: 76.6551 } };
    }

    const { lat, lon } = networkInfo.center;

    map = L.map('map', {
        center: [lat, lon],
        zoom: 14,
        zoomControl: false,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
    }).addTo(map);

    // Initialize layer groups
    layers.dijkstra = L.layerGroup().addTo(map);
    layers.ml       = L.layerGroup().addTo(map);
    layers.vehicles = L.layerGroup().addTo(map);
    layers.markers  = L.layerGroup().addTo(map);
    layers.stations = L.layerGroup().addTo(map);

    // Map click sets DESTINATION only
    map.on('click', onMapClick);

    if (networkInfo.bounds) {
        const { sw, ne } = networkInfo.bounds;
        map.fitBounds([[sw.lat, sw.lon], [ne.lat, ne.lon]]);
    }
}

function onMapClick(e) {
    const { lat, lng } = e.latlng;
    setDestination(lat, lng, `${lat.toFixed(5)}, ${lng.toFixed(5)}`);
}

function setDestination(lat, lng, label) {
    dstLatLng = { lat, lng };
    if (dstMarker) map.removeLayer(dstMarker);
    dstMarker = L.circleMarker([lat, lng], {
        radius:      11,
        fillColor:   '#a78bfa',
        fillOpacity: 0.9,
        color:       '#fff',
        weight:      2.5,
    }).addTo(layers.markers);
    dstMarker.bindPopup('<b>Destination</b>').openPopup();

    // Update UI indicator
    const ind = document.getElementById('dst-indicator');
    document.getElementById('dst-label').textContent = label;
    ind.classList.add('has-dst');
}

// ── Stations ───────────────────────────────────────────────────────────────
async function loadStations() {
    try {
        const res = await fetch('/api/stations');
        stationsData = await res.json();
        populateStationDropdown(stationsData);
        renderStationMarkers(stationsData);
    } catch (e) {
        console.error('Failed to load stations:', e);
    }
}

function populateStationDropdown(stations) {
    const sel = document.getElementById('sel-station');

    // Group by type
    const groups = { hospital: [], fire: [], police: [] };
    stations.forEach(s => groups[s.type]?.push(s));

    const labels = { hospital: '🏥 Hospitals', fire: '🚒 Fire Stations', police: '🚔 Police Stations' };

    for (const [type, list] of Object.entries(groups)) {
        if (!list.length) continue;
        const og = document.createElement('optgroup');
        og.label = labels[type];
        list.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.icon} ${s.name}`;
            opt.dataset.lat = s.lat;
            opt.dataset.lon = s.lon;
            opt.dataset.edge = s.edge_id || '';
            og.appendChild(opt);
        });
        sel.appendChild(og);
    }
}

function renderStationMarkers(stations) {
    if (!layers.stations) return;
    layers.stations.clearLayers();

    stations.forEach(s => {
        const icon = L.divIcon({
            html: `<div class="station-marker-icon" title="${s.name}">${s.icon}</div>`,
            className: '',
            iconSize:  [30, 30],
            iconAnchor:[15, 15],
        });

        const marker = L.marker([s.lat, s.lon], { icon })
            .bindPopup(`<b>${s.icon} ${s.name}</b><br><small>${s.type.charAt(0).toUpperCase()+s.type.slice(1)}</small>`);

        marker.on('click', () => {
            // Auto-select this station in dropdown
            const sel = document.getElementById('sel-station');
            sel.value = s.id;
            onStationChange(s.id);
        });

        layers.stations.addLayer(marker);
    });
}

function onStationChange(stationId) {
    if (!stationId) {
        selectedStation = null;
        return;
    }
    selectedStation = stationsData.find(s => s.id === stationId) || null;
    if (selectedStation && map) {
        map.flyTo([selectedStation.lat, selectedStation.lon], 15, { duration: 1.2 });
    }
}

// ── Condition Chips ────────────────────────────────────────────────────────
function selectChip(el) {
    const group = el.dataset.group;
    const val   = el.dataset.val;

    // De-activate all chips in the group
    document.querySelectorAll(`.chip[data-group="${group}"]`).forEach(c => {
        c.className = 'chip';
    });

    // Activate clicked chip
    el.classList.add(`active-${group.replace('_', '-')}-${val}`);

    // Update conditions state
    conditions[group] = val;
}

// ── Status ─────────────────────────────────────────────────────────────────
async function loadStatus() {
    try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        updateStatusUI(data);
    } catch (e) {
        console.error('Status fetch failed', e);
    }
}

function updateStatusUI(data) {
    setPill('status-sumo',  data.sumo_running    ? 'online' : 'offline', 'SUMO');
    setPill('status-model', data.model_loaded     ? 'online' : 'offline', 'Model');
    setPill('status-traci', data.traci_connected  ? 'online' : 'offline', 'TraCI');

    document.getElementById('stat-vehicles').textContent = data.vehicle_count || 0;
    document.getElementById('stat-step').textContent     = data.sim_step      || 0;
    document.getElementById('nav-sim-step').textContent  = data.sim_step      || 0;
}

function setPill(id, state, label) {
    const el = document.getElementById(id);
    if (!el) return;  // element may have been removed
    el.className = `status-pill ${state}`;
    const labelEl = el.querySelector('.label');
    if (labelEl) labelEl.textContent = label;
}

// ── Socket.IO Events ───────────────────────────────────────────────────────
function bindSocketEvents() {
    socket.on('connect', () => console.log('Socket connected'));

    socket.on('status', (data) => {
        updateStatusUI(data);
        if (data.sumo_running) hideSumoBanner();
        if (data.error) showToast('⚠️ TraCI: ' + data.error, false);
    });

    socket.on('sim_update', (data) => {
        document.getElementById('stat-step').textContent    = data.step;
        document.getElementById('nav-sim-step').textContent = data.step;
        document.getElementById('stat-vehicles').textContent = data.count;
        updateVehicles(data.vehicles);
    });

    socket.on('new_mission', (mission) => {
        addMissionToLog(mission);
        document.getElementById('stat-missions').textContent =
            parseInt(document.getElementById('stat-missions').textContent || '0') + 1;
        updateMissionsBadge();
    });

    socket.on('training_status', (data) => {
        handleTrainingStatus(data);
    });
}

function startStatusPoller() {
    setInterval(loadStatus, 5000);
}

// ── Vehicle Rendering ──────────────────────────────────────────────────────
const vehicleMarkers = {};

function updateVehicles(vehicles) {
    if (!layers.vehicles) return;

    const seen = new Set(Object.keys(vehicles));

    for (const vid in vehicleMarkers) {
        if (!seen.has(vid)) {
            layers.vehicles.removeLayer(vehicleMarkers[vid]);
            delete vehicleMarkers[vid];
        }
    }

    for (const [vid, v] of Object.entries(vehicles)) {
        const pos = [v.lat, v.lon];
        if (vehicleMarkers[vid]) {
            vehicleMarkers[vid].setLatLng(pos);
        } else {
            const isEmergency = vid.startsWith('ev_') || v.type === 'emergency';
            vehicleMarkers[vid] = L.circleMarker(pos, {
                radius:      isEmergency ? 8 : 5,
                fillColor:   isEmergency ? '#f43f5e' : '#f59e0b',
                fillOpacity: 0.9,
                color:       '#fff',
                weight:      1.5,
            });
            vehicleMarkers[vid].bindTooltip(
                `<b>${vid}</b><br>${v.speed} km/h<br>${v.edge}`,
                { className: 'leaflet-tooltip-dark' }
            );
            layers.vehicles.addLayer(vehicleMarkers[vid]);
        }
    }
}

// ── Launch & Compute ───────────────────────────────────────────────────────
async function launchAndCompute() {
    // Validate inputs
    if (!selectedStation) {
        showToast('⚠️ Please select an origin station.', false);
        return;
    }
    if (!dstLatLng) {
        showToast('⚠️ Please click the map to set a destination.', false);
        return;
    }

    const btn = document.getElementById('btn-launch');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Launching SUMO…';

    showSumoBanner('Launching SUMO-GUI and connecting TraCI…');

    try {
        // Step 1: Launch SUMO with current conditions
        const launchRes = await fetch('/api/launch_sumo', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(conditions),
        });
        const launchData = await launchRes.json();

        if (!launchRes.ok || !launchData.ok) {
            showToast('❌ SUMO launch failed: ' + (launchData.error || launchData.message), false);
            hideSumoBanner();
            return;
        }

        showToast('✅ ' + launchData.message, true);
        showSumoBanner('SUMO running — computing path…');

        // Step 2: Wait a moment then compute route
        // (TraCI connects in background; route can still compute offline)
        await new Promise(r => setTimeout(r, 1500));
        await computeRoute();

    } catch (e) {
        showToast('❌ Launch error: ' + e.message, false);
        hideSumoBanner();
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Launch &amp; Compute Path`;
    }
}

async function computeRoute() {
    if (!selectedStation || !dstLatLng) return;

    // Build request body
    const body = {
        conditions: { ...conditions },
    };

    // Use edge_id from station if resolved, else fall back to lat/lon
    if (selectedStation.edge_id) {
        body.src_edge = selectedStation.edge_id;
    } else {
        body.src_lat = selectedStation.lat;
        body.src_lon = selectedStation.lon;
    }

    body.dst_lat = dstLatLng.lat;
    body.dst_lon = dstLatLng.lng;

    try {
        const res  = await fetch('/api/route', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(body),
        });
        const data = await res.json();

        if (!res.ok || data.error) {
            showToast('❌ ' + (data.error || 'Route failed'), false);
        } else {
            lastResult = data;
            renderRoutes(data);
            renderResults(data);
            hideSumoBanner();
        }
    } catch (e) {
        showToast('Route failed: ' + e.message, false);
    }
}

// ── Manual SUMO Connect (fallback) ─────────────────────────────────────────
async function manualConnect() {
    const btn = document.getElementById('btn-manual-connect');
    btn.disabled = true;
    try {
        const res  = await fetch('/api/connect_sumo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body:   JSON.stringify({ port: 8813 }),
        });
        const data = await res.json();
        showToast(data.ok ? '✅ ' + data.message : '❌ ' + data.message, data.ok);
        if (data.ok) await loadStatus();
    } catch (e) {
        showToast('❌ Connection error: ' + e.message, false);
    } finally {
        btn.disabled = false;
    }
}

// ── Route Rendering ────────────────────────────────────────────────────────
function renderRoutes(data) {
    if (layers.dijkstra) layers.dijkstra.clearLayers();
    if (layers.ml)       layers.ml.clearLayers();

    const dijkCoords = data.dijkstra?.coordinates || [];
    const mlCoords   = data.ml?.coordinates       || [];

    if (dijkCoords.length > 1) {
        const lls = dijkCoords.map(c => [c.lat, c.lon]);
        L.polyline(lls, {
            color:     '#22d3ee',
            weight:    5,
            opacity:   0.85,
            dashArray: '8 4',
            lineCap:   'round',
        }).bindTooltip('Dijkstra Route').addTo(layers.dijkstra);
    }

    if (mlCoords.length > 1) {
        const lls = mlCoords.map(c => [c.lat, c.lon]);
        L.polyline(lls, {
            color:   '#a78bfa',
            weight:  5,
            opacity: 0.9,
            lineCap: 'round',
        }).bindTooltip('ML Route').addTo(layers.ml);
        map.fitBounds(L.polyline(lls).getBounds(), { padding: [40, 40] });
    } else if (dijkCoords.length > 1) {
        const lls = dijkCoords.map(c => [c.lat, c.lon]);
        map.fitBounds(L.polyline(lls).getBounds(), { padding: [40, 40] });
    }
}

function renderResults(data) {
    const panel = document.getElementById('card-results');
    panel.style.display = 'block';
    panel.classList.add('fade-in');

    const d = data.dijkstra || {};
    const m = data.ml       || {};

    // Both travel_times are now ML-estimated (fair comparison)
    document.getElementById('dijk-time').textContent     = d.travel_time          ?? '—';
    document.getElementById('dijk-freeflow').textContent = d.travel_time_freeflow  ?? '—';
    document.getElementById('dijk-dist').textContent     = d.distance              ?? '—';
    document.getElementById('dijk-hops').textContent     = d.hops                  ?? '—';

    document.getElementById('ml-time').textContent  = m.travel_time ?? '—';
    document.getElementById('ml-dist').textContent  = m.distance    ?? '—';
    document.getElementById('ml-hops').textContent  = m.hops        ?? '—';

    const imp   = data.improvement_pct || 0;
    const badge = document.getElementById('improvement-badge');
    if (imp > 1) {
        badge.textContent = `ML saves ${imp.toFixed(1)}%`;
        badge.className   = 'improvement-badge positive';
    } else if (imp >= 0) {
        badge.textContent = `Paths equivalent (~${imp.toFixed(1)}%)`;
        badge.className   = 'improvement-badge neutral';
    } else {
        // Should rarely happen — means Dijkstra found a lower ML-cost path
        badge.textContent = `Dijkstra better by ${Math.abs(imp).toFixed(1)}%`;
        badge.className   = 'improvement-badge negative';
    }

    document.querySelector('.dijkstra-card').style.boxShadow =
        data.winner === 'dijkstra' ? '0 0 0 2px #22d3ee' : '';
    document.querySelector('.ml-card').style.boxShadow =
        data.winner === 'ml' ? '0 0 0 2px #a78bfa' : '';
}


// ── Mission Log ────────────────────────────────────────────────────────────
function addMissionToLog(mission) {
    const list  = document.getElementById('mission-list');
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    const imp   = mission.result?.improvement_pct || 0;
    const isML  = imp > 0;
    const cond  = mission.conditions || {};

    const condBadges = Object.entries({
        [cond.weather || '']:       cond.weather,
        [cond.time_of_day || '']:   cond.time_of_day,
        [cond.incident_type || '']: cond.incident_type,
    }).filter(([k]) => k).map(([k]) =>
        `<span class="cond-pill">${k}</span>`
    ).join('');

    const item = document.createElement('div');
    item.className = 'mission-item fade-in';
    item.innerHTML = `
        <span class="mission-id">${mission.id}</span>
        <div class="mission-info">
            <div class="mission-win ${isML ? 'ml-win' : 'dijk-win'}">
                ${isML ? `ML saves ${imp}%` : `Dijkstra wins (${Math.abs(imp)}%)`}
            </div>
            <div class="mission-detail">
                ${mission.result?.dijkstra?.travel_time ?? '?'}s → ${mission.result?.ml?.travel_time ?? '?'}s
            </div>
            <div class="cond-pills">${condBadges}</div>
        </div>
    `;

    item.addEventListener('click', () => {
        if (mission.result) {
            renderRoutes(mission.result);
            renderResults(mission.result);
        }
    });

    list.prepend(item);
}

function updateMissionsBadge() {
    const items = document.querySelectorAll('.mission-item').length;
    document.getElementById('missions-count-badge').textContent = items;
    document.getElementById('stat-missions').textContent        = items;
}

// ── Layer Toggles ──────────────────────────────────────────────────────────
function toggleLayer(name, visible) {
    const layer = layers[name];
    if (!layer) return;
    if (visible) {
        if (!map.hasLayer(layer)) map.addLayer(layer);
    } else {
        map.removeLayer(layer);
    }
}

// ── SUMO Banner ────────────────────────────────────────────────────────────
let bannerTimer = null;

function showSumoBanner(msg) {
    const el = document.getElementById('sumo-banner');
    document.getElementById('sumo-banner-msg').textContent = msg;
    el.classList.remove('hidden');
    if (bannerTimer) clearTimeout(bannerTimer);
}

function hideSumoBanner() {
    const el = document.getElementById('sumo-banner');
    el.classList.add('hidden');
}

// ── Training ───────────────────────────────────────────────────────────────
async function triggerTraining() {
    const btn = document.getElementById('btn-train');
    btn.disabled = true;
    showTrainingToast('Training LightGBM model...');

    const res  = await fetch('/api/train', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) {
        hideTrainingToast();
        btn.disabled = false;
    }
}

function handleTrainingStatus(data) {
    if (data.status === 'started') {
        showTrainingToast('Training in progress...');
    } else if (data.status === 'complete') {
        hideTrainingToast();
        document.getElementById('btn-train').disabled = false;
        showToast('✅ Model trained! MAE: ' + (data.meta?.mae || '?') + 's', true);
        setPill('status-model', 'online', 'Model');
    } else if (data.status === 'error') {
        hideTrainingToast();
        document.getElementById('btn-train').disabled = false;
        showToast('❌ Training failed: ' + data.message, false);
    }
}

// ── Toast Helpers ──────────────────────────────────────────────────────────
let toastTimer = null;

function showToast(msg, success = true) {
    const t  = document.getElementById('training-toast');
    const sp = t.querySelector('.spinner');
    sp.style.display = 'none';
    document.getElementById('toast-msg').textContent = msg;
    t.style.display = 'flex';
    t.style.borderColor = success ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(hideTrainingToast, 4000);
}

function showTrainingToast(msg) {
    const t  = document.getElementById('training-toast');
    t.querySelector('.spinner').style.display = 'block';
    document.getElementById('toast-msg').textContent = msg;
    t.style.display  = 'flex';
    t.style.borderColor = 'var(--accent-blue)';
}

function hideTrainingToast() {
    document.getElementById('training-toast').style.display = 'none';
}

// ── Modal ──────────────────────────────────────────────────────────────────
function closeModal() {
    document.getElementById('modal-backdrop').style.display = 'none';
}
