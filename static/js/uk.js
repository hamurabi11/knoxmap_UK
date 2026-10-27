// KnoxMap UK: the UK mode's panel (web/uk.py). The top bar's "UK mode" button shows
// it (remembered in this browser; KnoxMap.bat --uk starts with it on) and centres
// the map on London.
//
// 0. Data: which of the extra datasets are here, and where to download the rest.
// 1. Set up an area: setup_area.py for the box drawn on the map (the same
//    south/west/north/east fields the Generate button reads).
// 2. Run the UK chain: the map's uk.json saved from the form, then
//    uk_pipeline.py; its STEP lines move the bar. When it is done the map
//    opens in the usual cards (openMap, app.js) for Compile and Install.
//    The source map takes KnoxMap's own settings from the panel above
//    (settingsForRequest, app.js), as Generate does.
(function () {
  const $ = id => document.getElementById(id);
  if (!$('ukCard')) return;

  const LONDON = [51.5074, -0.1278];
  const KEY = 'knoxmapUkMode';
  let dataLoaded = false;

  function stored() {
    try { return localStorage.getItem(KEY); } catch (_) { return null; }
  }

  function setMode(on, centre) {
    $('ukCard').hidden = !on;
    $('ukModeBtn').setAttribute('aria-pressed', on ? 'true' : 'false');
    document.body.classList.toggle('uk-mode', on);
    try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (_) { /* a private window: not remembered */ }
    if (on && centre && typeof map !== 'undefined') map.setView(LONDON, 12);
    if (on && !dataLoaded) loadData();
  }

  $('ukModeBtn').addEventListener('click', () => setMode($('ukCard').hidden, true));

  async function loadData() {
    let d;
    try { d = await (await fetch('/api/uk/data')).json(); } catch (_) { return; }
    dataLoaded = true;
    const have = d.datasets.filter(x => x.have).length;
    $('ukDataSummary').textContent = `${have} of ${d.datasets.length} downloads here`
      + (d.epcCouncils.length ? `; EPC for ${d.epcCouncils.join(', ')}` : '');
    $('ukDataList').innerHTML = '';
    for (const x of d.datasets) {
      const li = document.createElement('li');
      const mark = document.createElement('span');
      mark.className = x.have ? 'ok' : 'missing';
      mark.textContent = x.have ? '✓ ' : '✗ ';
      const name = document.createElement('b');
      name.textContent = x.name;
      li.append(mark, name, document.createTextNode(` — ${x.gives}. `));
      const a = document.createElement('a');
      a.href = x.url;
      a.textContent = x.have ? 'Download page' : 'Download';
      a.addEventListener('click', ev => {       // the app window opens it in the browser
        ev.preventDefault();
        post('/api/uk/open', { key: x.key }).catch(() => window.open(x.url, '_blank'));
      });
      if (x.install) {
        // the tile pack: fetched, checked and installed by the app itself
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-small';
        btn.textContent = x.have ? 'Update it' : 'Download and install';
        btn.addEventListener('click', installPack);
        li.append(document.createTextNode(x.have ? `Installed: ${x.note}. ` : 'Not installed. '), btn,
                  document.createTextNode(' '), a);
        a.textContent = 'releases';
      } else if (x.have) {
        li.append(document.createTextNode(`Here: ${x.note}. `), a, document.createTextNode(' for a newer release.'));
      } else {
        li.append(a, document.createTextNode(` and save it in ${x.save}.`));
      }
      $('ukDataList').append(li);
    }
    $('ukAutoList').replaceChildren(...d.automatic.map(x => {
      const li = document.createElement('li');
      const b = document.createElement('b');
      b.textContent = x.name;
      li.append('• ', b, ` — ${x.gives}`);
      return li;
    }));
  }

  async function installPack() {
    try {
      const r = await post('/api/uk/pack/install', {});
      $('ukLogPanel').open = true;
      poll(r.job, 'UK tile pack', () => { dataLoaded = false; loadData(); });
    } catch (err) {
      fx.toast('bad', 'Could not install the UK tile pack', err.message, 8000);
    }
  }

  $('ukOpenRaw').addEventListener('click', () => post('/api/uk/open', { folder: 'raw' }).catch(() => {}));
  $('ukOpenInspire').addEventListener('click', () => post('/api/uk/open', { folder: 'inspire' }).catch(() => {}));

  let watching = null;      // the map whose job is polled
  let pollTimer = null;

  async function post(url, body) {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function settingsFromForm() {
    const wc = $('ukWorldCell').value.split(',').map(v => v.trim()).filter(Boolean);
    return {
      source: $('ukSource').value.trim() || null,
      area: $('ukArea').value.trim() || null,
      roads: $('ukRoads').value,
      // "lidar:<front anchoring>" or "knoxmap:<KnoxMap's building alignment>"
      buildings: $('ukBuildings').value.split(':')[0],
      front_anchor: { auto: null, on: true, off: false }[$('ukBuildings').value.split(':')[1]] ?? null,
      building_alignment: $('ukBuildings').value.startsWith('knoxmap:') ? $('ukBuildings').value.split(':')[1] : null,
      uk_assets: $('ukAssets').checked,
      lanes: $('ukLanes').checked,
      rows_any_angle: $('ukRowsAnyAngle').checked,
      rooms: $('ukRooms').checked,
      dormers: $('ukDormers').checked,
      inspire_attrs: $('ukInspire').checked,
      compile: $('ukCompile').checked,
      streets: $('ukStreets').value,
      world_cell: wc.length === 2 ? wc : null,
      backend: $('ukBackend').value || null,
    };
  }

  function fillForm(s) {
    if (!s) return;
    if (s.source) $('ukSource').value = s.source;
    if (s.area) $('ukArea').value = s.area;
    $('ukRoads').value = s.roads || 'graph';
    $('ukBuildings').value = s.buildings === 'knoxmap' ? `knoxmap:${s.building_alignment || 'smart'}`
      : `lidar:${s.front_anchor === true ? 'on' : s.front_anchor === false ? 'off' : 'auto'}`;
    $('ukAssets').checked = s.uk_assets !== false;
    $('ukLanes').checked = !!s.lanes;
    $('ukRowsAnyAngle').checked = !!s.rows_any_angle;
    $('ukRooms').checked = s.rooms !== false;
    $('ukDormers').checked = !!s.dormers;
    $('ukInspire').checked = s.inspire_attrs !== false;
    $('ukCompile').checked = !!s.compile;
    $('ukStreets').value = (s.streets || []).join(', ');
    $('ukWorldCell').value = s.world_cell ? s.world_cell.join(', ') : '';
    $('ukBackend').value = s.backend || '';
  }

  async function loadState(mapName) {
    const data = await (await fetch(`/api/uk/state?map=${encodeURIComponent(mapName || '')}`)).json();
    // folder names as options' values, never as markup
    const options = (list, values) => list.replaceChildren(...values.map(v => {
      const o = document.createElement('option');
      o.value = v;
      return o;
    }));
    options($('ukMapList'), data.maps.map(m => m.map));
    options($('ukAreaList'), data.areas.map(a => a.area));
    $('ukAreas').textContent = data.areas.length
      ? 'Areas: ' + data.areas.map(a => `${a.area} (${a.data === 'osm_lidar' ? 'OSM + LiDAR' : 'full'}`
          + (a.council ? `, ${a.council}` : '') + ')').join('; ')
      : 'No areas set up yet.';
    if (data.settings) fillForm(data.settings);
    return data;
  }

  function showJob(job, label) {
    const running = job.state === 'running' || job.state === 'stopping';
    $('ukStopBtn').hidden = !running;
    $('ukStopBtn').disabled = job.state === 'stopping';
    $('ukRunBtn').disabled = $('ukSetupBtn').disabled = running;
    $('ukBar').style.width = job.n ? `${Math.round(100 * Math.max(0, job.k - (running ? 1 : 0)) / job.n)}%`
      : (running ? '5%' : '0');
    const where = job.step ? ` — ${job.step}${job.n ? ` (${job.k} of ${job.n})` : ''}` : '';
    $('ukStatus').textContent = {
      idle: '', running: `${label}${where}…`, stopping: 'Stopping…',
      done: `${label}: done.`, failed: `${label}: failed${where}. See the output.`,
      stopped: `${label}: stopped${where}. Run it again to carry on from there.`,
    }[job.state] || job.state;
    $('ukLog').textContent = (job.lines || []).join('\n');
    if (job.state === 'failed') $('ukLogPanel').open = true;
  }

  function poll(mapName, label, onDone) {
    watching = mapName;
    clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      let job;
      try {
        job = await (await fetch(`/api/uk/job?map=${encodeURIComponent(mapName)}`)).json();
      } catch (_) { return; }
      showJob(job, label);
      if (job.state !== 'running' && job.state !== 'stopping') {
        clearInterval(pollTimer);
        pollTimer = null;
        if (job.state === 'done' && onDone) onDone();
      }
    }, 2000);
  }

  $('ukSetupBtn').addEventListener('click', async () => {
    const name = $('ukSourceName').value.trim();
    const box = ['south', 'west', 'north', 'east'].map(k => $(k).value);
    if (!name) { fx.toast('bad', 'UK mode', 'Give the source map a name', 6000); return; }
    if (box.some(v => v === '')) { fx.toast('bad', 'UK mode', 'Draw the area on the map first', 6000); return; }
    try {
      const r = await post('/api/uk/setup', {
        name, area: $('ukAreaName').value.trim(), data: $('ukLevel').value, rotate: $('ukRotate').value,
        south: box[0], west: box[1], north: box[2], east: box[3],
        // KnoxMap's own settings, from the Settings panel (as Generate sends them)
        settings: typeof settingsForRequest === 'function' ? settingsForRequest() : undefined,
      });
      // The run form ready for this area's map
      $('ukSource').value = name;
      $('ukArea').value = r.area;
      if (!$('ukMapName').value) $('ukMapName').value = `${name}_graph`;
      poll(name, `Setting up ${r.area}`, () => loadState($('ukMapName').value));
    } catch (err) {
      fx.toast('bad', 'Could not set up the area', err.message, 8000);
    }
  });

  $('ukRunBtn').addEventListener('click', async () => {
    const mapName = $('ukMapName').value.trim();
    if (!mapName) { fx.toast('bad', 'UK mode', 'Which map? (the source map\'s name with _graph)', 6000); return; }
    try {
      await post('/api/uk/settings', { map: mapName, settings: settingsFromForm() });
      await post('/api/uk/run', { map: mapName, force: $('ukForce').checked });
      poll(mapName, `UK chain on ${mapName}`, () => { if (typeof openMap === 'function') openMap(mapName); });
    } catch (err) {
      fx.toast('bad', 'Could not run the UK chain', err.message, 8000);
    }
  });

  $('ukStopBtn').addEventListener('click', async () => {
    if (!watching) return;
    $('ukStopBtn').disabled = true;
    try { await post('/api/uk/stop', { map: watching }); } catch (_) { /* the job's state says */ }
  });

  $('ukMapName').addEventListener('change', async () => {
    const data = await loadState($('ukMapName').value.trim());
    if (data.job && data.job.state !== 'idle') {
      showJob(data.job, `UK chain on ${$('ukMapName').value.trim()}`);
      if (data.job.state === 'running') poll($('ukMapName').value.trim(), `UK chain on ${$('ukMapName').value.trim()}`);
    }
  });

  loadState('');
  // On at start when KnoxMap.bat --uk says so, else as it was left in this browser;
  // on, the map starts over London.
  const startOn = $('ukModeBtn').dataset.startOn === '1' || stored() === '1';
  setMode(startOn, startOn);
})();
