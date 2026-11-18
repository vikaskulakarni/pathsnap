import os
import time
import json
import logging
import numpy as np
import folium
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocessing import extract_candidates_and_features_vectorized
from hmm_decoder import HMMRouteDecoder, load_trip_assets

# Configure rich console logging for evaluators
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [PathSnap API] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap: Distributed Map-Matching & Route Reconstruction API",
    description="Real-time inference API for noisy GPS map matching to road networks using Gradient Boosted Models and HMM Viterbi Decoding",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_BASE = r'Student folder\Student folder'
TEST_DIR = os.path.join(DATA_BASE, 'student_test')
TRAIN_DIR = os.path.join(DATA_BASE, 'train')
ARTIFACTS_DIR = 'artifacts'

# Load model and decoder eagerly
def init_assets():
    model_path = os.path.join(ARTIFACTS_DIR, 'final_tuned_model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, 'random_forest.joblib')
    logger.info(f"Loading trained model artifact from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Initializing HMM Viterbi Decoder (sigma_z=10.0m, beta=15.0m)...")
    decoder = HMMRouteDecoder(sigma_z=10.0, beta=15.0, connectivity_bonus=3.0, disconnect_penalty=-12.0)
    logger.info("PathSnap API initialization complete. Ready to receive inference requests.")
    return model, decoder

MODEL, DECODER = init_assets()

class PredictRequest(BaseModel):
    trip_id: str

@app.get("/health")
def health_check():
    return {
        "status": "ONLINE",
        "service": "PathSnap ML Inference Backend",
        "model_loaded": MODEL is not None,
        "available_test_trips": sorted(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else []
    }

@app.get("/trips")
def list_trips():
    test_trips = sorted(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else []
    return {"evaluation_trips": test_trips}

@app.post("/predict")
def predict_route(request: PredictRequest):
    trip_id = request.trip_id.strip()
    logger.info(f"===> INCOMING REQUEST: Predict Route for Trip ID: '{trip_id}'")

    trip_dir = os.path.join(TEST_DIR, trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(TRAIN_DIR, trip_id)
    if not os.path.exists(trip_dir):
        logger.error(f"Trip ID '{trip_id}' not found in dataset directories.")
        raise HTTPException(status_code=404, detail=f"Trip ID '{trip_id}' not found.")

    t_start = time.time()

    # 1. Load assets
    logger.info(f"Loading raw road-network files and GPS track for trip '{trip_id}'...")
    nodes, arcs, track, route = load_trip_assets(trip_dir, trip_id)
    T = len(track)
    logger.info(f"Loaded {T} GPS observations, {len(nodes)} nodes, {len(arcs)} road segments (arcs).")

    # 2. Candidate generation & feature extraction
    logger.info("Executing spatial KD-Tree candidate generation & feature extraction...")
    t_feat_0 = time.time()
    feats, meta, miss_rate = extract_candidates_and_features_vectorized(trip_dir, trip_id, k_candidates=5)
    t_feat = time.time() - t_feat_0
    logger.info(f"Extracted features for {len(feats)} candidate pairs in {t_feat:.3f}s (Miss rate: {miss_rate*100:.2f}%).")

    # 3. Model inference
    logger.info("Running trained ML model probability inference...")
    t_inf_0 = time.time()
    probs = MODEL.predict_proba(feats)[:, 1]
    t_inf = time.time() - t_inf_0
    logger.info(f"Model prediction completed in {t_inf:.3f}s ({len(feats)/t_inf:.0f} cands/sec).")

    meta['pred_prob'] = probs
    meta['dist'] = feats['dist_to_segment'].values

    # 4. HMM Viterbi Sequential Route Reconstruction
    logger.info("Constructing trellis and running Viterbi decoding...")
    t_hmm_0 = time.time()
    candidates_per_step = []
    probs_per_step = []
    dists_per_step = []
    clf_top1_arcs = []
    true_arcs_list = []

    for obs_idx, group in meta.groupby('obs_idx'):
        c_arcs = group['candidate_arc'].values
        c_probs = group['pred_prob'].values
        c_dists = group['dist'].values

        candidates_per_step.append(c_arcs)
        probs_per_step.append(c_probs)
        dists_per_step.append(c_dists)
        clf_top1_arcs.append(c_arcs[np.argmax(c_probs)])
        true_arcs_list.append(group['true_arc'].iloc[0])

    clf_top1_arcs = np.array(clf_top1_arcs)
    true_arcs = np.array(true_arcs_list)

    viterbi_route = DECODER.decode_trip(track, arcs, candidates_per_step, probs_per_step, dists_per_step)
    t_hmm = time.time() - t_hmm_0
    logger.info(f"Viterbi optimal path decoded in {t_hmm:.3f}s.")

    # 5. Accuracy & Continuity calculation (Server-side ground truth)
    clf_acc = float(np.mean(clf_top1_arcs == true_arcs) * 100) if route is not None else None
    hmm_acc = float(np.mean(viterbi_route == true_arcs) * 100) if route is not None else None

    def calc_continuity(seq):
        if len(seq) <= 1:
            return 100.0
        c = sum(1 for i in range(len(seq)-1) if seq[i] == seq[i+1] or arcs[seq[i], 1] == arcs[seq[i+1], 0])
        return float(c / (len(seq)-1) * 100)

    clf_cont = calc_continuity(clf_top1_arcs)
    hmm_cont = calc_continuity(viterbi_route)

    t_total = time.time() - t_start
    logger.info(f"Trip '{trip_id}' Results -> Classifier Acc: {clf_acc:.2f}%, HMM Acc: {hmm_acc:.2f}%, Continuity: {hmm_cont:.2f}%")
    logger.info(f"Total processing latency: {t_total:.3f}s. Request completed successfully.")

    # Generate Folium map HTML
    center_lat = float(track[:, 1].mean())
    center_lon = float(track[:, 0].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')

    # GPS points
    gps_layer = folium.FeatureGroup(name="Raw GPS Track (Blue Markers)")
    step = max(1, len(track) // 500)
    for pt in track[::step]:
        folium.CircleMarker(
            location=[float(pt[1]), float(pt[0])],
            radius=3, color='#1f77b4', fill=True, fill_color='#1f77b4', fill_opacity=0.75
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # Ground truth
    if route is not None:
        gt_layer = folium.FeatureGroup(name="Ground Truth Route (Green)")
        gt_pts = []
        for a in route:
            gt_pts.extend([[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
                           [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]])
        folium.PolyLine(locations=gt_pts, color='#2ca02c', weight=5, opacity=0.8).add_to(gt_layer)
        gt_layer.add_to(fmap)

    # Predicted route
    pred_layer = folium.FeatureGroup(name="PathSnap Reconstructed Route (Red)")
    pred_pts = []
    # Deduplicate consecutive identical arcs
    u_arcs = [viterbi_route[0]] if len(viterbi_route) > 0 else []
    for a in viterbi_route[1:]:
        if a != u_arcs[-1]:
            u_arcs.append(a)
    for a in u_arcs:
        pred_pts.extend([[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])],
                         [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]])
    folium.PolyLine(locations=pred_pts, color='#d62728', weight=3.5, dash_array='5, 5', opacity=0.9).add_to(pred_layer)
    pred_layer.add_to(fmap)

    folium.LayerControl().add_to(fmap)
    map_html = fmap.get_root().render()

    return {
        "status": "SUCCESS",
        "trip_id": trip_id,
        "observations_count": T,
        "classifier_top1_accuracy_pct": round(clf_acc, 2) if clf_acc is not None else "N/A",
        "hmm_route_accuracy_pct": round(hmm_acc, 2) if hmm_acc is not None else "N/A",
        "classifier_continuity_pct": round(clf_cont, 2),
        "hmm_continuity_pct": round(hmm_cont, 2),
        "candidate_miss_rate_pct": round(miss_rate * 100, 2),
        "total_latency_seconds": round(t_total, 3),
        "predicted_route_arcs_sample": [int(x) for x in viterbi_route[:50]],
        "map_html": map_html
    }

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """
    Evaluator User Interface (runs in browser on Laptop 2 or Laptop 1).
    """
    trips = sorted(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else []
    trips_options = "".join([f"<option value='{t}'>Trip {t}</option>" for t in trips])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PathSnap: Map-Matching Evaluator Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }}
            .card {{ border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.06); border: none; }}
            .metric-badge {{ font-size: 1.8rem; font-weight: 700; color: #1e3a8a; }}
            .metric-title {{ font-size: 0.85rem; text-transform: uppercase; color: #64748b; font-weight: 600; }}
            #map-container {{ height: 580px; width: 100%; border-radius: 8px; overflow: hidden; border: 1px solid #cbd5e1; }}
            iframe {{ width: 100%; height: 100%; border: none; }}
        </style>
    </head>
    <body class="p-3">
        <div class="container-fluid">
            <header class="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom">
                <div>
                    <h3 class="fw-bold text-primary mb-0">PathSnap: Trajectory Reconstruction System</h3>
                    <small class="text-muted">Distributed Evaluation Interface | Laptop 2 Client &bull; Laptop 1 Backend</small>
                </div>
                <div class="badge bg-success p-2 fs-6">Backend: Connected</div>
            </header>

            <div class="row g-3">
                <!-- Control Panel -->
                <div class="col-md-4">
                    <div class="card p-3 mb-3">
                        <h5 class="fw-bold mb-3">Select Evaluation Trip</h5>
                        <p class="text-muted small">Choose any of the 10 reserved evaluation trips to send inference request to the API backend:</p>
                        <select id="tripSelector" class="form-select form-select-lg mb-3">
                            {trips_options}
                        </select>
                        <button id="runBtn" class="btn btn-primary btn-lg w-100 fw-bold" onclick="runEvaluation()">
                            Run Map-Matching Pipeline
                        </button>
                    </div>

                    <!-- Metrics Badges -->
                    <div class="card p-3 mb-3">
                        <h5 class="fw-bold mb-3">Evaluation Metrics</h5>
                        <div class="row text-center g-2">
                            <div class="col-6">
                                <div class="p-2 border rounded bg-white">
                                    <div class="metric-title">HMM Continuity</div>
                                    <div id="mContinuity" class="metric-badge text-success">--</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="p-2 border rounded bg-white">
                                    <div class="metric-title">Top-1 Accuracy</div>
                                    <div id="mAccuracy" class="metric-badge text-primary">--</div>
                                </div>
                            </div>
                            <div class="col-6 mt-2">
                                <div class="p-2 border rounded bg-white">
                                    <div class="metric-title">GPS Observations</div>
                                    <div id="mObs" class="metric-badge text-secondary">--</div>
                                </div>
                            </div>
                            <div class="col-6 mt-2">
                                <div class="p-2 border rounded bg-white">
                                    <div class="metric-title">Total Latency</div>
                                    <div id="mLatency" class="metric-badge text-dark">--</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Pipeline Details -->
                    <div class="card p-3">
                        <h6 class="fw-bold text-muted mb-2">Live System Status</h6>
                        <ul class="list-group list-group-flush small" id="statusList">
                            <li class="list-group-item d-flex justify-content-between"><span>Model Architecture:</span> <strong class="text-primary">Gradient Boosted + HMM</strong></li>
                            <li class="list-group-item d-flex justify-content-between"><span>Candidate Miss Rate:</span> <span id="mMissRate">--</span></li>
                            <li class="list-group-item d-flex justify-content-between"><span>Classifier Baseline Acc:</span> <span id="mClfAcc">--</span></li>
                            <li class="list-group-item d-flex justify-content-between"><span>Classifier Continuity:</span> <span id="mClfCont">--</span></li>
                        </ul>
                    </div>
                </div>

                <!-- Visualization Panel -->
                <div class="col-md-8">
                    <div class="card p-3 h-100">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h5 class="fw-bold mb-0">Interactive Geographical Visualization</h5>
                            <div>
                                <span class="badge bg-primary">Blue: Raw GPS</span>
                                <span class="badge bg-success">Green: Ground Truth</span>
                                <span class="badge bg-danger">Red: PathSnap Route</span>
                            </div>
                        </div>
                        <div id="map-container">
                            <div class="h-100 d-flex justify-content-center align-items-center text-muted">
                                Select a trip and click "Run Map-Matching Pipeline" to load map.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function runEvaluation() {{
                const tripId = document.getElementById('tripSelector').value;
                const runBtn = document.getElementById('runBtn');
                runBtn.disabled = true;
                runBtn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing Pipeline...';

                try {{
                    const response = await fetch('/predict', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ trip_id: tripId }})
                    }});
                    const data = await response.json();

                    if (data.status === 'SUCCESS') {{
                        document.getElementById('mContinuity').innerText = data.hmm_continuity_pct + '%';
                        document.getElementById('mAccuracy').innerText = data.hmm_route_accuracy_pct + '%';
                        document.getElementById('mObs').innerText = data.observations_count;
                        document.getElementById('mLatency').innerText = data.total_latency_seconds + 's';
                        document.getElementById('mMissRate').innerText = data.candidate_miss_rate_pct + '%';
                        document.getElementById('mClfAcc').innerText = data.classifier_top1_accuracy_pct + '%';
                        document.getElementById('mClfCont').innerText = data.classifier_continuity_pct + '%';

                        // Render folium map in iframe
                        const container = document.getElementById('map-container');
                        container.innerHTML = '';
                        const iframe = document.createElement('iframe');
                        container.appendChild(iframe);
                        iframe.contentWindow.document.open();
                        iframe.contentWindow.document.write(data.map_html);
                        iframe.contentWindow.document.close();
                    }} else {{
                        alert('Error: ' + JSON.stringify(data));
                    }}
                }} catch (err) {{
                    alert('API Error: ' + err.message);
                }} finally {{
                    runBtn.disabled = false;
                    runBtn.innerHTML = 'Run Map-Matching Pipeline';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
