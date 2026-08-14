# 🏢 SEEFLOOR— Building Risk Analyzer for Small-Scale Buildings in Iloilo City

### Thesis Project | DPWH / National Building Code of the Philippines

---

## 📋 What this does

This is a **Flask web application** that implements the IPO (Input-Process-Output) model for building fire risk analysis based on the DPWH National Building Code of the Philippines (PD 1096) and RA 9514.

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Flask server

```bash
python app.py
```

### 3. Open in browser

```
http://localhost:5000
```

---

## 🔁 IPO Flow

### INPUT

- Floor plan image (uploaded, display only — for now)
- Room data per room:
  - Room name
  - Room type (classroom, lab, kitchen, etc.)
  - Floor level
  - Distance to nearest exit (meters)
  - Adjacent room type

### PROCESS (Algorithms)

1. **Hazard Index (HI) Computation**
   - `HI = (distance / max_distance) × room_weight × floor_multiplier × adjacency_factor`
   - Room weights based on occupancy type hazard
   - Floor multiplier: higher floors = harder to evacuate
   - Adjacency factor: being next to dangerous rooms raises risk

2. **Risk-Level Classification**
   - 🟢 Green: HI < 0.5 → Low Risk
   - 🟠 Orange: 0.5 ≤ HI < 0.9 → Moderate Risk
   - 🔴 Red: HI ≥ 0.9 → High Risk

3. **High-Risk Zone Cluster Detection**
   - Groups adjacent high-risk rooms on the same floor

4. **Rule-Based Recommendations**
   - Based on DPWH max evacuation distance (30m non-sprinklered, 45m sprinklered)
   - References PD 1185, NBC Section 805, RA 9514

### OUTPUT

- Floor-Level Risk Indicator (per floor HI + bar chart)
- Building Risk Index (average HI across all rooms)
- Room-by-room risk table
- Identified High-Risk Clusters
- Safety Recommendations

---

## 📁 Project Structure

```
floorplan_app/
├── app.py                  # Flask backend + algorithms
├── requirements.txt
├── templates/
│   └── index.html          # Main UI
└── static/
    ├── css/style.css       # Styling
    ├── js/app.js           # Frontend logic
    └── uploads/            # Uploaded floor plans stored here
```

---

## 🏛️ DPWH Standards Referenced

- **PD 1096** — National Building Code of the Philippines
- **PD 1185** — Fire Code of the Philippines
- **RA 9514** — Revised Fire Code of the Philippines (2008)
- Max travel distance to exit: **30m** (non-sprinklered), **45m** (sprinklered)
- Minimum corridor width: **1.2m**

---

## 🔮 Future Development

- [ ] OpenAI / image AI integration to auto-scan floor plan images
- [ ] Graph construction (nodes & edges) for pathfinding
- [ ] Dynamic multi-layer floorplan view
- [ ] PDF report export
- [ ] Database for saving building records
