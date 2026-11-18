import os
import json
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import joblib

def to_meters(lon, lat, lon0, lat0):
    m_lat = 111132.92
    m_lon = 111412.84 * np.cos(np.radians(lat0))
    x = (lon - lon0) * m_lon
    y = (lat - lat0) * m_lat
    return x, y

def load_trip_data(trip_dir, trip_id):
    nodes = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.nodes"))
    arcs = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.arcs"), dtype=int)
    track = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.track"))
    route_p = os.path.join(trip_dir, f"{trip_id}.route")
    route = np.loadtxt(route_p, dtype=int) if os.path.exists(route_p) else None
    return nodes, arcs, track, route

def extract_candidates_and_features_vectorized(trip_dir, trip_id, k_candidates=5):
    nodes, arcs, track, route = load_trip_data(trip_dir, trip_id)
    T = len(track)
    lat0, lon0 = track[:, 1].mean(), track[:, 0].mean()
    tx, ty = to_meters(track[:, 0], track[:, 1], lon0, lat0)
    nx, ny = to_meters(nodes[:, 0], nodes[:, 1], lon0, lat0)

    # Arc endpoints & lengths
    u, v = arcs[:, 0], arcs[:, 1]
    ax, ay = nx[u], ny[u]
    bx, by = nx[v], ny[v]
    arc_dx, arc_dy = bx - ax, by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_bearing = np.arctan2(arc_dy, arc_dx)

    # Ground truth arcs if route available
    true_arcs = None
    if route is not None:
        ru, rv = arcs[route, 0], arcs[route, 1]
        rax, ray = nx[ru], ny[ru]
        rbx, rby = nx[rv], ny[rv]
        rdx, rdy = rbx - rax, rby - ray
        rl2 = np.maximum(rdx*rdx + rdy*rdy, 1e-9)

        px, py = tx[:, None], ty[:, None]
        rt_proj = np.clip(((px - rax[None, :])*rdx[None, :] + (py - ray[None, :])*rdy[None, :]) / rl2[None, :], 0.0, 1.0)
        r_dists = np.hypot(px - (rax[None, :] + rt_proj * rdx[None, :]), py - (ray[None, :] + rt_proj * rdy[None, :]))
        best_r = np.argmin(r_dists, axis=1)
        true_arcs = route[best_r]

    # Spatial KDTree with 3-point sampling per arc
    N = len(arcs)
    sample_pts = np.vstack([
        np.column_stack([ax, ay]),
        np.column_stack([(ax + bx)*0.5, (ay + by)*0.5]),
        np.column_stack([bx, by])
    ])
    arc_indices = np.concatenate([np.arange(N), np.arange(N), np.arange(N)])
    tree = cKDTree(sample_pts)

    # GPS dynamics
    gps_dt = np.diff(track[:, 2], prepend=track[0, 2])
    gps_dt = np.where(gps_dt <= 0, 1.0, gps_dt)
    gps_dx = np.diff(tx, prepend=tx[0])
    gps_dy = np.diff(ty, prepend=ty[0])
    gps_speed = np.hypot(gps_dx, gps_dy) / gps_dt
    gps_bearing = np.arctan2(gps_dy, gps_dx)

    # Query nearest candidate arcs
    _, nn_idx = tree.query(np.column_stack([tx, ty]), k=30)
    cand_arcs = arc_indices[nn_idx] # (T, 30)

    # Compute distances to all 30 candidate arcs
    c_ax, c_ay = ax[cand_arcs], ay[cand_arcs]
    c_dx, c_dy = arc_dx[cand_arcs], arc_dy[cand_arcs]
    c_l2 = np.maximum(c_dx*c_dx + c_dy*c_dy, 1e-9)

    proj = np.clip(((tx[:, None] - c_ax)*c_dx + (ty[:, None] - c_ay)*c_dy) / c_l2, 0.0, 1.0)
    proj_x = c_ax + proj * c_dx
    proj_y = c_ay + proj * c_dy
    cand_dists = np.hypot(tx[:, None] - proj_x, ty[:, None] - proj_y)

    # Select top k_candidates
    top_order = np.argsort(cand_dists, axis=1)[:, :k_candidates]
    top_arcs = np.take_along_axis(cand_arcs, top_order, axis=1)        # (T, K)
    top_dists = np.take_along_axis(cand_dists, top_order, axis=1)      # (T, K)
    top_projs = np.take_along_axis(proj, top_order, axis=1)            # (T, K)

    # Feature extraction (fully vectorized)
    c_ax_top = ax[top_arcs]
    c_ay_top = ay[top_arcs]
    c_bx_top = bx[top_arcs]
    c_by_top = by[top_arcs]

    px = tx[:, None]
    py = ty[:, None]

    f_dist_seg = top_dists
    f_dist_start = np.hypot(px - c_ax_top, py - c_ay_top)
    f_dist_end = np.hypot(px - c_bx_top, py - c_by_top)
    f_proj_ratio = top_projs
    f_arc_len = arc_len[top_arcs]

    # Bearing difference
    diff_b = np.abs(gps_bearing[:, None] - arc_bearing[top_arcs])
    f_bearing_diff = np.abs((diff_b + np.pi) % (2 * np.pi) - np.pi)

    f_gps_speed = np.tile(gps_speed[:, None], (1, k_candidates))
    f_dist_rank = np.tile(np.arange(k_candidates)[None, :], (T, 1))
    f_dist_diff_min = top_dists - top_dists[:, 0:1]
    f_norm_dist = top_dists / (np.mean(top_dists, axis=1, keepdims=True) + 1e-5)

    # Stack features into (T * K, num_features)
    feature_mats = [
        f_dist_seg.ravel(),
        f_dist_start.ravel(),
        f_dist_end.ravel(),
        f_proj_ratio.ravel(),
        f_arc_len.ravel(),
        f_bearing_diff.ravel(),
        f_gps_speed.ravel(),
        f_dist_rank.ravel(),
        f_dist_diff_min.ravel(),
        f_norm_dist.ravel()
    ]
    X = np.column_stack(feature_mats)

    feature_names = [
        'dist_to_segment', 'dist_to_start', 'dist_to_end', 'proj_ratio',
        'arc_length', 'bearing_diff', 'gps_speed', 'dist_rank',
        'dist_diff_min', 'norm_dist'
    ]

    obs_indices = np.repeat(np.arange(T), k_candidates)
    flat_cand_arcs = top_arcs.ravel()

    # Labels and Candidate Miss Rate
    y = np.zeros(T * k_candidates, dtype=int)
    miss_rate = 0.0
    if true_arcs is not None:
        matches = (top_arcs == true_arcs[:, None])
        y = matches.astype(int).ravel()
        # Candidate miss: none of the top_arcs matched true_arcs
        miss_count = np.sum(~np.any(matches, axis=1))
        miss_rate = float(miss_count / max(1, T))

    df_meta = pd.DataFrame({
        'obs_idx': obs_indices,
        'candidate_arc': flat_cand_arcs,
        'label': y,
        'true_arc': np.repeat(true_arcs if true_arcs is not None else -1, k_candidates)
    })

    return pd.DataFrame(X, columns=feature_names), df_meta, miss_rate

def build_dataset(base_dir=r'Student folder\Student folder', n_train_trips=25):
    print("Building dataset from raw files...")
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'student_test')

    all_train_trips = sorted(os.listdir(train_dir))
    test_trips = sorted(os.listdir(test_dir))

    selected_train = all_train_trips[:n_train_trips]
    print(f"Processing {len(selected_train)} training trips (Vectorized)...")

    X_train_list, y_train_list, groups_train_list = [], [], []
    train_misses = []

    t0 = time.time()
    for i, tid in enumerate(selected_train):
        t_dir = os.path.join(train_dir, tid)
        feats, meta, miss_rate = extract_candidates_and_features_vectorized(t_dir, tid)
        X_train_list.append(feats)
        y_train_list.append(meta['label'].values)
        groups_train_list.append(np.full(len(feats), i))
        train_misses.append(miss_rate)
        if (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{len(selected_train)} train trips (Miss rate: {np.mean(train_misses)*100:.2f}%)")

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)
    groups_train = np.concatenate(groups_train_list)
    t1 = time.time()

    print(f"Train dataset created in {t1 - t0:.2f}s: {X_train.shape[0]} rows, {X_train.shape[1]} features.")
    print(f"Class balance: Positive={np.sum(y_train==1)} ({np.mean(y_train==1)*100:.1f}%), Negative={np.sum(y_train==0)}")
    print(f"Average Candidate Miss Rate on Train: {np.mean(train_misses)*100:.2f}%")

    print(f"\nProcessing {len(test_trips)} student test trips...")
    X_test_list, y_test_list, test_meta_list = [], [], []
    test_misses = []

    for i, tid in enumerate(test_trips):
        t_dir = os.path.join(test_dir, tid)
        feats, meta, miss_rate = extract_candidates_and_features_vectorized(t_dir, tid)
        meta['trip_id'] = tid
        X_test_list.append(feats)
        y_test_list.append(meta['label'].values)
        test_meta_list.append(meta)
        test_misses.append(miss_rate)

    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = np.concatenate(y_test_list)
    df_test_meta = pd.concat(test_meta_list, ignore_index=True)

    print(f"Student Test dataset created: {X_test.shape[0]} rows.")
    print(f"Average Candidate Miss Rate on Student Test: {np.mean(test_misses)*100:.2f}%")

    os.makedirs('data', exist_ok=True)
    joblib.dump(X_train, 'data/X_train.joblib')
    joblib.dump(y_train, 'data/y_train.joblib')
    joblib.dump(groups_train, 'data/groups_train.joblib')
    joblib.dump(X_test, 'data/X_test.joblib')
    joblib.dump(y_test, 'data/y_test.joblib')
    joblib.dump(df_test_meta, 'data/test_meta.joblib')

    with open('data/miss_rates.json', 'w') as f:
        json.dump({
            'train_miss_rate': float(np.mean(train_misses)),
            'test_miss_rate': float(np.mean(test_misses))
        }, f, indent=2)

    print("Data successfully generated and cached in 'data/'!")
    return X_train, y_train, groups_train, X_test, y_test, df_test_meta

if __name__ == '__main__':
    build_dataset()
