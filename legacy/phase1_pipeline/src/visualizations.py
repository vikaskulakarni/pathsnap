import os
import numpy as np
import folium
import joblib

def generate_interactive_route_map(trip_dir, trip_id, predicted_arcs, output_path=None):
    nodes = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.nodes"))
    arcs = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.arcs"), dtype=int)
    track = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.track"))
    route_p = os.path.join(trip_dir, f"{trip_id}.route")
    route = np.loadtxt(route_p, dtype=int) if os.path.exists(route_p) else None

    center_lat = float(track[:, 1].mean())
    center_lon = float(track[:, 0].mean())

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='OpenStreetMap')

    # Layer 1: Raw GPS Points
    gps_layer = folium.FeatureGroup(name="Raw GPS Track (Blue Markers)")
    step = max(1, len(track) // 600) # sample up to 600 points for smooth browser rendering
    for pt in track[::step]:
        folium.CircleMarker(
            location=[float(pt[1]), float(pt[0])],
            radius=3,
            color='#1f77b4',
            fill=True,
            fill_color='#1f77b4',
            fill_opacity=0.75,
            tooltip=f"Time: {pt[2]:.1f}s"
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Layer 2: Ground Truth Route
    if route is not None:
        gt_layer = folium.FeatureGroup(name="Ground Truth Route (Green)")
        gt_coords = []
        for arc_idx in route:
            u, v = arcs[arc_idx]
            gt_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
            gt_coords.append([float(nodes[v, 1]), float(nodes[v, 0])])
        folium.PolyLine(
            locations=gt_coords,
            color='#2ca02c',
            weight=5,
            opacity=0.8,
            tooltip="Ground Truth Path"
        ).add_to(gt_layer)
        gt_layer.add_to(m)

    # Layer 3: Predicted Route (HMM / ML Pipeline)
    pred_layer = folium.FeatureGroup(name="PathSnap Predicted Route (Red)")
    pred_coords = []
    # Deduplicate consecutive identical arcs for smooth polyline
    unique_pred_arcs = []
    for a in predicted_arcs:
        if not unique_pred_arcs or unique_pred_arcs[-1] != a:
            unique_pred_arcs.append(a)

    for arc_idx in unique_pred_arcs:
        u, v = arcs[arc_idx]
        pred_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
        pred_coords.append([float(nodes[v, 1]), float(nodes[v, 0])])

    folium.PolyLine(
        locations=pred_coords,
        color='#d62728',
        weight=3.5,
        dash_array='6, 6',
        opacity=0.9,
        tooltip="Predicted Route"
    ).add_to(pred_layer)
    pred_layer.add_to(m)

    folium.LayerControl().add_to(m)

    if output_path is None:
        output_path = os.path.join('plots', f"trip_{trip_id}_evaluation_map.html")
    m.save(output_path)
    print(f"Generated interactive map saved to {output_path}")
    return m

def generate_all_evaluation_maps(base_dir=r'Student folder\Student folder\student_test',
                                 artifacts_dir='artifacts', plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    from hmm_decoder import HMMRouteDecoder, load_trip_assets

    model = joblib.load(os.path.join(artifacts_dir, 'final_tuned_model.joblib'))
    X_test = joblib.load('data/X_test.joblib')
    df_test_meta = joblib.load('data/test_meta.joblib')
    df_test_meta['pred_prob'] = model.predict_proba(X_test)[:, 1]
    df_test_meta['dist'] = X_test['dist_to_segment'].values

    decoder = HMMRouteDecoder(sigma_z=10.0, beta=15.0, connectivity_bonus=3.0, disconnect_penalty=-12.0)

    # Generate maps for sample student test trips
    for tid in ['00000000', '00000007', '00000052']:
        trip_dir = os.path.join(base_dir, tid)
        nodes, arcs, track, route = load_trip_assets(trip_dir, tid)
        trip_meta = df_test_meta[df_test_meta['trip_id'] == tid].sort_values(['obs_idx', 'candidate_arc'])

        candidates_per_step = []
        probs_per_step = []
        dists_per_step = []
        for obs_idx, group in trip_meta.groupby('obs_idx'):
            candidates_per_step.append(group['candidate_arc'].values)
            probs_per_step.append(group['pred_prob'].values)
            dists_per_step.append(group['dist'].values)

        pred_path = decoder.decode_trip(track, arcs, candidates_per_step, probs_per_step, dists_per_step)
        map_out = os.path.join(plots_dir, f"trip_{tid}_interactive_map.html")
        generate_interactive_route_map(trip_dir, tid, pred_path, output_path=map_out)

if __name__ == '__main__':
    generate_all_evaluation_maps()
