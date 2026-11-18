import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def run_eda(base_dir=r'Student folder\Student folder', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'student_test')

    train_trips = sorted(os.listdir(train_dir))
    test_trips = sorted(os.listdir(test_dir))

    print(f"=== Dataset Inventory ===")
    print(f"Training Trips: {len(train_trips)}")
    print(f"Student Testing Trips: {len(test_trips)}")

    inventory = []
    
    # Sample trips to analyze in detail
    sampling_intervals = []
    trajectory_lengths = []
    speeds = []
    node_counts = []
    arc_counts = []
    route_lens = []

    m_lat = 111132.92

    for partition, trips, p_dir in [('train', train_trips, train_dir), ('student_test', test_trips, test_dir)]:
        for tid in trips:
            track_p = os.path.join(p_dir, tid, f"{tid}.track")
            nodes_p = os.path.join(p_dir, tid, f"{tid}.nodes")
            arcs_p = os.path.join(p_dir, tid, f"{tid}.arcs")
            route_p = os.path.join(p_dir, tid, f"{tid}.route")

            track_data = np.loadtxt(track_p)
            n_obs = len(track_data)
            trajectory_lengths.append(n_obs)

            # Timestamps
            dt = np.diff(track_data[:, 2])
            sampling_intervals.extend(dt[dt < 60]) # filter huge gaps for interval stats

            # Speeds
            lat0 = track_data[:, 1].mean()
            m_lon = 111412.84 * np.cos(np.radians(lat0))
            d_lon = np.diff(track_data[:, 0]) * m_lon
            d_lat = np.diff(track_data[:, 1]) * m_lat
            dist = np.hypot(d_lon, d_lat)
            valid_dt = np.maximum(dt, 0.1)
            spd = dist / valid_dt
            speeds.extend(spd[spd < 60]) # filter unrealistic spikes

            with open(nodes_p) as f:
                n_nodes = sum(1 for _ in f)
            with open(arcs_p) as f:
                n_arcs = sum(1 for _ in f)
            n_route = 0
            if os.path.exists(route_p):
                with open(route_p) as f:
                    n_route = sum(1 for _ in f)

            node_counts.append(n_nodes)
            arc_counts.append(n_arcs)
            route_lens.append(n_route)

            inventory.append({
                'trip_id': tid,
                'partition': partition,
                'num_obs': n_obs,
                'num_nodes': n_nodes,
                'num_arcs': n_arcs,
                'route_arcs': n_route,
                'duration_sec': track_data[-1, 2] - track_data[0, 2],
                'min_lon': track_data[:, 0].min(),
                'max_lon': track_data[:, 0].max(),
                'min_lat': track_data[:, 1].min(),
                'max_lat': track_data[:, 1].max()
            })

    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv(os.path.join(output_dir, 'dataset_inventory.csv'), index=False)
    print(f"Inventory saved. Total observations: {df_inv['num_obs'].sum()}")
    print(f"Mean observations per trip: {df_inv['num_obs'].mean():.1f} (std: {df_inv['num_obs'].std():.1f})")
    print(f"Min observations: {df_inv['num_obs'].min()}, Max observations: {df_inv['num_obs'].max()}")

    # 1. Plot Trajectory Lengths Distribution
    plt.figure(figsize=(9, 4.5))
    sns.histplot(df_inv['num_obs'], bins=25, kde=True, color='teal')
    plt.title('Distribution of Trajectory Lengths (GPS Observations per Trip)', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Observations', fontsize=11)
    plt.ylabel('Trip Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_length_dist.png'), dpi=300)
    plt.close()

    # 2. Plot Sampling Interval Distribution
    sampling_intervals = np.array(sampling_intervals)
    exact_1s = np.mean((sampling_intervals >= 0.99) & (sampling_intervals <= 1.01)) * 100
    print(f"Sampling interval ~1.0s: {exact_1s:.2f}% of steps")

    plt.figure(figsize=(9, 4.5))
    sns.histplot(sampling_intervals[sampling_intervals <= 5], bins=50, color='darkorange')
    plt.title(f'GPS Sampling Interval (dt) Distribution (Exact 1s: {exact_1s:.1f}%)', fontsize=13, fontweight='bold')
    plt.xlabel('Delta t (seconds)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sampling_interval_dist.png'), dpi=300)
    plt.close()

    # 3. Plot Speed Distribution
    speeds = np.array(speeds) * 3.6 # convert m/s to km/h
    plt.figure(figsize=(9, 4.5))
    sns.histplot(speeds, bins=40, kde=True, color='navy')
    plt.title('Empirical Speed Distribution (km/h)', fontsize=13, fontweight='bold')
    plt.xlabel('Speed (km/h)', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_distribution.png'), dpi=300)
    plt.close()

    # 4. Interactive Folium Map for Sample Trip 00000001
    print("Generating Folium Map for Trip 00000001...")
    sample_tid = '00000001'
    sample_dir = os.path.join(train_dir, sample_tid)
    nodes = np.loadtxt(os.path.join(sample_dir, f"{sample_tid}.nodes"))
    arcs = np.loadtxt(os.path.join(sample_dir, f"{sample_tid}.arcs"), dtype=int)
    route = np.loadtxt(os.path.join(sample_dir, f"{sample_tid}.route"), dtype=int)
    track = np.loadtxt(os.path.join(sample_dir, f"{sample_tid}.track"))

    center_lat = track[:, 1].mean()
    center_lon = track[:, 0].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')

    # Add Raw GPS points
    gps_feature_group = folium.FeatureGroup(name="Raw GPS Track (Blue Points)")
    for pt in track[::3]: # decimate slightly for faster rendering
        folium.CircleMarker(
            location=[pt[1], pt[0]],
            radius=3,
            color='#1f77b4',
            fill=True,
            fill_color='#1f77b4',
            fill_opacity=0.7,
            popup=f"Time: {pt[2]}s"
        ).add_to(gps_feature_group)
    gps_feature_group.add_to(m)

    # Add Ground Truth Route
    gt_coords = []
    for arc_idx in route:
        u, v = arcs[arc_idx]
        pt1 = [nodes[u, 1], nodes[u, 0]]
        pt2 = [nodes[v, 1], nodes[v, 0]]
        gt_coords.append(pt1)
        gt_coords.append(pt2)

    folium.PolyLine(
        locations=gt_coords,
        color='#2ca02c',
        weight=4,
        opacity=0.85,
        tooltip="Ground Truth Route (Green)"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    folium_path = os.path.join(output_dir, 'sample_trip_00000001_map.html')
    m.save(folium_path)
    print(f"Folium Map saved to {folium_path}")

    # Summary statistics text report
    eda_summary = {
        'total_train_trips': len(train_trips),
        'total_test_trips': len(test_trips),
        'total_observations': int(df_inv['num_obs'].sum()),
        'mean_observations': float(df_inv['num_obs'].mean()),
        'median_observations': float(df_inv['num_obs'].median()),
        'exact_1s_pct': float(exact_1s),
        'mean_speed_kmh': float(np.mean(speeds)),
        'p95_speed_kmh': float(np.percentile(speeds, 95)),
        'mean_road_nodes': float(np.mean(node_counts)),
        'mean_road_arcs': float(np.mean(arc_counts))
    }
    with open(os.path.join(output_dir, 'eda_summary.json'), 'w') as f:
        json.dump(eda_summary, f, indent=2)

    print("EDA completed successfully!")
    return eda_summary

if __name__ == '__main__':
    run_eda()
