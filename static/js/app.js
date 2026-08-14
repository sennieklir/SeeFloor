const ROOM_TYPES = [
  "classroom", "laboratory", "office", "restroom", "corridor",
  "stairwell", "storage", "kitchen", "electrical_room",
  "server_room", "conference", "lobby", "other"
];

let uploadedFloorplanPath = null;
let rowCount = 0;






const uploadZone = document.getElementById('upload-zone');
const floorplanInput = document.getElementById('floorplan-input');
const uploadContent = document.getElementById('upload-content');
const uploadPreview = document.getElementById('upload-preview');

uploadZone.addEventListener('click', () => floorplanInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.style.borderColor = '#9ccce2'; });
uploadZone.addEventListener('dragleave', () => uploadZone.style.borderColor = '');
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.style.borderColor = '';
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

floorplanInput.addEventListener('change', () => {
  if (floorplanInput.files[0]) handleFile(floorplanInput.files[0]);
});

function handleFile(file) {
  const formData = new FormData();
  formData.append('floorplan', file);


  const reader = new FileReader();
  reader.onload = e => {
    uploadContent.classList.add('hidden');
    uploadPreview.src = e.target.result;
    uploadPreview.classList.remove('hidden');
    uploadZone.classList.add('has-file');
  };
  reader.readAsDataURL(file);

  
  fetch('/upload', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      if (data.success) uploadedFloorplanPath = data.filepath;
    })
    .catch(err => console.error('Upload failed:', err));
}







function makeTypeOptions(selected = '') {
  return ROOM_TYPES.map(t =>
    `<option value="${t}" ${t === selected ? 'selected' : ''}>${formatLabel(t)}</option>`
  ).join('');
}

function formatLabel(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function addRoom(data = {}) {
  rowCount++;
  const tbody = document.getElementById('room-tbody');
  const tr = document.createElement('tr');
  tr.dataset.id = rowCount;
  tr.innerHTML = `
    <td><input type="text" placeholder="nya room" value="${data.room_name || ''}" /></td>
    <td>
      <select>${makeTypeOptions(data.room_type || 'classroom')}</select>
    </td>
    <td><input type="number" min="1" max="50" value="${data.floor_level || 1}" style="width:70px" /></td>
    <td><input type="number" min="0" step="0.1" value="${data.distance_to_exit || ''}" placeholder="nya meters" style="width:100px" /></td>
    <td>
      <select>${makeTypeOptions(data.adjacency || 'other')}</select>
    </td>
    <td><button class="btn-remove" onclick="removeRow(this)" title="Remove">×</button></td>
  `;
  tbody.appendChild(tr);
}

function removeRow(btn) {
  btn.closest('tr').remove();
}

function getTableData() {
  const rows = document.querySelectorAll('#room-tbody tr');
  const rooms = [];
  let valid = true;

  rows.forEach(tr => {
    const inputs = tr.querySelectorAll('input');
    const selects = tr.querySelectorAll('select');
    const name = inputs[0].value.trim();
    const floor = parseInt(inputs[1].value);
    const dist = parseFloat(inputs[2].value);
    const roomType = selects[0].value;
    const adjacency = selects[1].value;

    if (!name || isNaN(dist) || isNaN(floor)) { valid = false; return; }
    rooms.push({ room_name: name, room_type: roomType, floor_level: floor, distance_to_exit: dist, adjacency });
  });

  return { rooms, valid };
}

















async function runAnalysis() {
  const { rooms, valid } = getTableData();

  if (!rooms.length) return showAlert('Please add at least one room.');
  if (!valid) return showAlert('Please fill in all room fields properly.');

  const buildingName = document.getElementById('building-name').value.trim() || 'Unnamed Building';


  setStep(2);
  document.getElementById('section-input').classList.add('hidden');
  document.getElementById('section-process').classList.remove('hidden');
  document.getElementById('section-output').classList.add('hidden');

  const logEl = document.getElementById('process-log');
  const steps = [
    'Parsing room data...',
    'Computing Hazard Index per room...',
    'Classifying risk levels (Green/Orange/Red)...',
    'Detecting high-risk clusters...',
    'Generating DPWH-based recommendations...',
    'Building output report...'
  ];

  logEl.innerHTML = steps.map((s, i) => `<li id="plog-${i}">${s}</li>`).join('');


  for (let i = 0; i < steps.length; i++) {
    await delay(350);
    if (i > 0) document.getElementById(`plog-${i-1}`).className = 'done';
    document.getElementById(`plog-${i}`).className = 'active';
  }

  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ building_name: buildingName, rooms, floorplan_path: uploadedFloorplanPath })
    });
    const data = await res.json();
    await delay(400);
    document.getElementById(`plog-${steps.length-1}`).className = 'done';
    await delay(300);
    showResults(data);
  } catch (err) {
    showAlert('Analysis failed. Make sure the Flask server is running.');
    resetApp();
  }
}

function showResults(data) {
  document.getElementById('section-process').classList.add('hidden');
  document.getElementById('section-output').classList.remove('hidden');
  setStep(3);

 
  const bri = data.building_risk_index;
  document.getElementById('bri-value').textContent = bri.toFixed(4);
  document.getElementById('total-rooms').textContent = data.rooms.length;
  document.getElementById('high-risk-count').textContent = data.rooms.filter(r => r.risk_color === 'red').length;
  document.getElementById('cluster-count').textContent = data.clusters.length;

  const briLabel = document.getElementById('bri-label');
  briLabel.textContent = data.building_risk_label;
  briLabel.className = `summary-badge badge-${data.rooms.length ? colorBadge(bri) : 'green'}`;


  if (data.floorplan_path) {
    document.getElementById('floorplan-card').style.display = '';
    document.getElementById('result-floorplan').src = data.floorplan_path;
  }


  const floorEl = document.getElementById('floor-risk-list');
  floorEl.innerHTML = '';
  Object.entries(data.floor_risk).sort((a,b) => a[0]-b[0]).forEach(([fl, info]) => {
    const pct = Math.min(info.index / 1.5 * 100, 100).toFixed(1);
    const color = hiColor(info.index);
    floorEl.innerHTML += `
      <div class="floor-risk-item">
        <div class="floor-number">Floor ${fl}</div>
        <div class="floor-bar-wrap">
          <div class="floor-bar-bg">
            <div class="floor-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
        </div>
        <div class="floor-hi" style="color:${color}">${info.index.toFixed(4)}</div>
        <span class="risk-pill risk-${hiClass(info.index)}">${info.label}</span>
      </div>`;
  });


  const tbody = document.getElementById('result-tbody');
  tbody.innerHTML = '';
  data.rooms.forEach(room => {
    const color = hiColor(room.hazard_index);
    tbody.innerHTML += `
      <tr>
        <td>${room.room_name}</td>
        <td>${formatLabel(room.room_type)}</td>
        <td>${room.floor_level}</td>
        <td>${room.distance_to_exit} m</td>
        <td><span class="hi-val" style="color:${color}">${room.hazard_index.toFixed(4)}</span></td>
        <td><span class="risk-pill risk-${room.risk_color}">${room.risk_label}</span></td>
      </tr>`;
  });

  
  const clusterEl = document.getElementById('cluster-list');
  if (data.clusters.length === 0) {
    clusterEl.innerHTML = '<p style="color:#00b894;font-weight:600">✅ No high-risk clusters detected.</p>';
  } else {
    clusterEl.innerHTML = data.clusters.map(c => `
      <div class="cluster-item">
        <div class="cluster-title">🔥 Cluster ${c.cluster_id} — Floor ${c.floor}</div>
        <div class="cluster-rooms">Rooms: ${c.rooms.join(', ')} &nbsp;|&nbsp; Avg. HI: <strong>${c.avg_hi}</strong></div>
      </div>`).join('');
  }


  const recEl = document.getElementById('recommendations-list');
  recEl.innerHTML = data.recommendations.map(r => `
    <div class="rec-item ${r.priority.toLowerCase()}">
      <div class="rec-icon">${r.icon}</div>
      <div class="rec-body">
        <div class="rec-priority priority-${r.priority.toLowerCase()}">${r.priority}</div>
        <div class="rec-message">${r.message}</div>
      </div>
    </div>`).join('');


  document.getElementById('section-output').scrollIntoView({ behavior: 'smooth' });
}

function resetApp() {
  document.getElementById('section-output').classList.add('hidden');
  document.getElementById('section-process').classList.add('hidden');
  document.getElementById('section-input').classList.remove('hidden');
  setStep(1);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}



function setStep(n) {
  [1,2,3].forEach(i => {
    const el = document.getElementById(`step-dot-${i}`);
    el.className = 'step' + (i === n ? ' active' : i < n ? ' done' : '');
  });
}

function hiColor(hi) {
  if (hi < 0.5) return '#00b894';
  if (hi < 0.9) return '#e17055';
  return '#d63031';
}

function hiClass(hi) {
  if (hi < 0.5) return 'green';
  if (hi < 0.9) return 'orange';
  return 'red';
}

function colorBadge(bri) {
  if (bri < 0.5) return 'green';
  if (bri < 0.9) return 'orange';
  return 'red';
}

function formatLabel(str) {
  return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function showAlert(msg) {
  alert(msg);
}


addRoom(); 
