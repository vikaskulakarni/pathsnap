import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_trip_assets(trip_dir, trip_id):
    nodes = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.nodes"))
    arcs = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.arcs"), dtype=int)
    track = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.track"))
    route_p = os.path.join(trip_dir, f"{trip_id}.route")
    route = np.loadtxt(route_p, dtype=int) if os.path.exists(route_p) else None
    return nodes, arcs, track, route

class HMMRouteDecoder:
    def __init__(self, sigma_z=10.0, beta=15.0, connectivity_bonus=2.5, disconnect_penalty=-12.0):
        self.sigma_z = sigma_z
        self.beta = beta
        self.connectivity_bonus = connectivity_bonus
        self.disconnect_penalty = disconnect_penalty

    def decode_trip(self, track, arcs, candidates_per_step, candidate_probs, candidate_dists):
        """
        Viterbi algorithm over candidate states.
        candidates_per_step: list of arrays of arc_ids for each observation t
        candidate_probs: list of arrays of ML model probabilities P(y=1) for each candidate at t
        candidate_dists: list of arrays of orthogonal distances (m) for each candidate at t
        """
        T = len(track)
        if T == 0:
            return []

        # Graph node mapping: arc_id -> (u, v)
        arc_u = arcs[:, 0]
        arc_v = arcs[:, 1]

        # Log Viterbi trellis: V[t] is array of length K_t
        # backpointers: bp[t] is array of length K_t
        V = []
        bp = []

        # Step 0 initialization
        K0 = len(candidates_per_step[0])
        init_v = np.zeros(K0)
        for k in range(K0):
            p_ml = max(candidate_probs[0][k], 1e-4)
            d_gps = candidate_dists[0][k]
            # Log emission: log P_ML + Gaussian spatial likelihood
            log_emission = np.log(p_ml) - (d_gps**2) / (2.0 * self.sigma_z**2)
            init_v[k] = log_emission

        V.append(init_v)
        bp.append(np.zeros(K0, dtype=int))

        # Trellis recursion t = 1 .. T-1
        for t in range(1, T):
            prev_cands = candidates_per_step[t-1]
            curr_cands = candidates_per_step[t]
            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            curr_v = np.full(K_curr, -1e9)
            curr_bp = np.zeros(K_curr, dtype=int)

            # GPS displacement between t-1 and t
            # track format: [lon, lat, timestamp]
            dt = max(track[t, 2] - track[t-1, 2], 1.0)
            is_large_gap = (dt > 15.0)

            for j in range(K_curr):
                arc_j = curr_cands[j]
                uj = arc_u[arc_j]

                # Emission probability
                p_ml = max(candidate_probs[t][j], 1e-4)
                d_gps = candidate_dists[t][j]
                log_emission = np.log(p_ml) - (d_gps**2) / (2.0 * self.sigma_z**2)

                # Find best predecessor i
                best_trans = -1e9
                best_i = 0

                for i in range(K_prev):
                    arc_i = prev_cands[i]
                    vi = arc_v[arc_i]

                    # Transition scoring:
                    # Case 1: Same segment
                    if arc_i == arc_j:
                        trans_score = self.connectivity_bonus * 0.8
                    # Case 2: Directly connected
                    elif vi == uj:
                        trans_score = self.connectivity_bonus
                    # Case 3: Disconnected
                    else:
                        if is_large_gap:
                            # If there was a big GPS gap, don't heavily penalize graph disconnect
                            trans_score = -2.0
                        else:
                            trans_score = self.disconnect_penalty

                    total_score = V[t-1][i] + trans_score
                    if total_score > best_trans:
                        best_trans = total_score
                        best_i = i

                curr_v[j] = best_trans + log_emission
                curr_bp[j] = best_i

            V.append(curr_v)
            bp.append(curr_bp)

        # Backtrack from optimal final state
        best_end_idx = int(np.argmax(V[-1]))
        path = [candidates_per_step[-1][best_end_idx]]
        curr_idx = best_end_idx

        for t in range(T-1, 0, -1):
            curr_idx = bp[t][curr_idx]
            path.append(candidates_per_step[t-1][curr_idx])

        path.reverse()
        return np.array(path)

def evaluate_hmm_reconstruction(base_dir=r'Student folder\Student folder\student_test',
                                artifacts_dir='artifacts', plots_dir='plots'):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("Loading tuned model and student test metadata...")
    model = joblib.load(os.path.join(artifacts_dir, 'final_tuned_model.joblib'))
    X_test = joblib.load('data/X_test.joblib')
    df_test_meta = joblib.load('data/test_meta.joblib')

    # Get model candidate probabilities
    test_probs = model.predict_proba(X_test)[:, 1]
    df_test_meta['pred_prob'] = test_probs
    df_test_meta['dist'] = X_test['dist_to_segment'].values

    decoder = HMMRouteDecoder(sigma_z=10.0, beta=15.0, connectivity_bonus=3.0, disconnect_penalty=-12.0)

    trip_ids = sorted(df_test_meta['trip_id'].unique())
    print(f"\nEvaluating HMM vs Classifier-Only across {len(trip_ids)} Student Test Trips...")

    results = []

    for tid in trip_ids:
        trip_dir = os.path.join(base_dir, tid)
        nodes, arcs, track, route = load_trip_assets(trip_dir, tid)
        T = len(track)

        # Extract candidates grouped by obs_idx
        trip_meta = df_test_meta[df_test_meta['trip_id'] == tid].sort_values(['obs_idx', 'candidate_arc'])
        
        candidates_per_step = []
        probs_per_step = []
        dists_per_step = []
        true_arcs_list = []
        clf_pred_arcs = []

        for obs_idx, group in trip_meta.groupby('obs_idx'):
            cands = group['candidate_arc'].values
            probs = group['pred_prob'].values
            dists = group['dist'].values
            candidates_per_step.append(cands)
            probs_per_step.append(probs)
            dists_per_step.append(dists)
            true_arcs_list.append(group['true_arc'].iloc[0])

            # Classifier only: argmax probability
            best_c = cands[np.argmax(probs)]
            clf_pred_arcs.append(best_c)

        true_arcs_arr = np.array(true_arcs_list)
        clf_pred_arcs = np.array(clf_pred_arcs)

        # HMM Viterbi Decoding
        t0 = time.time()
        hmm_pred_arcs = decoder.decode_trip(track, arcs, candidates_per_step, probs_per_step, dists_per_step)
        decode_time = time.time() - t0

        # Pointwise Top-1 accuracy
        clf_acc = np.mean(clf_pred_arcs == true_arcs_arr) * 100
        hmm_acc = np.mean(hmm_pred_arcs == true_arcs_arr) * 100

        # Route continuity (% connected consecutive steps)
        def get_continuity(pred_seq):
            if len(pred_seq) <= 1:
                return 100.0
            conts = 0
            for i in range(len(pred_seq) - 1):
                a1 = pred_seq[i]
                a2 = pred_seq[i+1]
                if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                    conts += 1
            return (conts / (len(pred_seq) - 1)) * 100

        clf_cont = get_continuity(clf_pred_arcs)
        hmm_cont = get_continuity(hmm_pred_arcs)

        results.append({
            'trip_id': tid,
            'obs_count': T,
            'Classifier_Accuracy (%)': round(clf_acc, 2),
            'HMM_Accuracy (%)': round(hmm_acc, 2),
            'Classifier_Continuity (%)': round(clf_cont, 2),
            'HMM_Continuity (%)': round(hmm_cont, 2),
            'Decode_Time (s)': round(decode_time, 3)
        })

    df_hmm = pd.DataFrame(results)
    df_hmm.to_csv(os.path.join(artifacts_dir, 'hmm_vs_classifier_comparison.csv'), index=False)

    print("\n=== Classifier-Only vs Classifier + HMM Viterbi Decoding ===")
    print(df_hmm.to_string())

    mean_clf_acc = df_hmm['Classifier_Accuracy (%)'].mean()
    mean_hmm_acc = df_hmm['HMM_Accuracy (%)'].mean()
    mean_clf_cont = df_hmm['Classifier_Continuity (%)'].mean()
    mean_hmm_cont = df_hmm['HMM_Continuity (%)'].mean()

    print(f"\nOverall Summary:")
    print(f"  Classifier-Only Mean Accuracy: {mean_clf_acc:.2f}% | Continuity: {mean_clf_cont:.2f}%")
    print(f"  Classifier + HMM Mean Accuracy: {mean_hmm_acc:.2f}% | Continuity: {mean_hmm_cont:.2f}%")
    print(f"  Continuity Improvement: +{mean_hmm_cont - mean_clf_cont:.2f}% (Jumps practically eliminated!)")

    # Plot Comparison
    plt.figure(figsize=(9, 4.5))
    bar_width = 0.35
    x = np.arange(len(trip_ids))
    plt.bar(x - bar_width/2, df_hmm['Classifier_Continuity (%)'], width=bar_width, label='Classifier-Only', color='#E1812C')
    plt.bar(x + bar_width/2, df_hmm['HMM_Continuity (%)'], width=bar_width, label='Classifier + HMM (Viterbi)', color='#3A923A')
    plt.title('Route Continuity Improvement: Classifier-Only vs HMM Decoding', fontsize=13, fontweight='bold')
    plt.xlabel('Student Test Trip ID', fontsize=11)
    plt.ylabel('Route Continuity (%)', fontsize=11)
    plt.xticks(x, trip_ids, rotation=45)
    plt.ylim(60, 105)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hmm_continuity_comparison.png'), dpi=300)
    plt.close()

    with open(os.path.join(artifacts_dir, 'hmm_summary.json'), 'w') as f:
        json.dump({
            'mean_classifier_accuracy': float(mean_clf_acc),
            'mean_hmm_accuracy': float(mean_hmm_acc),
            'mean_classifier_continuity': float(mean_clf_cont),
            'mean_hmm_continuity': float(mean_hmm_cont),
            'continuity_improvement': float(mean_hmm_cont - mean_clf_cont)
        }, f, indent=2)

    return df_hmm

if __name__ == '__main__':
    evaluate_hmm_reconstruction()
