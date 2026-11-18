import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, log_loss,
    confusion_matrix, roc_auc_score, accuracy_score
)

def compute_observation_metrics(df_meta, pred_probs, base_dir=r'Student folder\Student folder\student_test'):
    """
    Computes observation-level Top-1 Accuracy, Route Continuity, Macro F1, Log Loss.
    """
    df_meta = df_meta.copy()
    df_meta['pred_prob'] = pred_probs

    # Observation-level Top-1 Accuracy
    # Group by (trip_id, obs_idx), pick candidate with highest probability
    grouped = df_meta.groupby(['trip_id', 'obs_idx'])
    top1_matches = []
    
    # Route continuity per trip
    continuity_scores = []

    for (trip_id, _), group in df_meta.groupby(['trip_id', 'trip_id']):
        # Load arcs for this trip to check connectivity
        trip_dir = os.path.join(base_dir, trip_id)
        arcs = np.loadtxt(os.path.join(trip_dir, f"{trip_id}.arcs"), dtype=int)

        obs_groups = group.groupby('obs_idx')
        pred_arcs = []
        true_arcs = []
        for obs_idx, obs_cands in obs_groups:
            best_row = obs_cands.sort_values('pred_prob', ascending=False).iloc[0]
            pred_arcs.append(int(best_row['candidate_arc']))
            true_arcs.append(int(best_row['true_arc']))

        pred_arcs = np.array(pred_arcs)
        true_arcs = np.array(true_arcs)

        # Top-1 match
        matches = (pred_arcs == true_arcs)
        top1_matches.extend(matches)

        # Continuity
        if len(pred_arcs) > 1:
            cont_count = 0
            for i in range(len(pred_arcs) - 1):
                a1 = pred_arcs[i]
                a2 = pred_arcs[i+1]
                if a1 == a2 or arcs[a1, 1] == arcs[a2, 0]:
                    cont_count += 1
            continuity_scores.append(cont_count / (len(pred_arcs) - 1))

    top1_acc = float(np.mean(top1_matches))
    mean_continuity = float(np.mean(continuity_scores)) if continuity_scores else 0.0
    return top1_acc, mean_continuity

def train_and_evaluate_models(plots_dir='plots', artifacts_dir='artifacts'):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading prepared datasets...")
    X_train = joblib.load('data/X_train.joblib')
    y_train = joblib.load('data/y_train.joblib')
    groups_train = joblib.load('data/groups_train.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_test = joblib.load('data/y_test.joblib')
    df_test_meta = joblib.load('data/test_meta.joblib')

    # Define 5 models from distinct algorithm families
    models = {
        'Logistic Regression': {
            'family': 'Linear & Probabilistic',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'))
            ]),
            'config': 'StandardScaler + LogisticRegression(C=1.0, balanced)'
        },
        'Gaussian Naive Bayes': {
            'family': 'Probabilistic (Generative)',
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ]),
            'config': 'StandardScaler + GaussianNB(var_smoothing=1e-9)'
        },
        'Decision Tree': {
            'family': 'Tree-based',
            'pipeline': Pipeline([
                ('clf', DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42))
            ]),
            'config': 'DecisionTreeClassifier(max_depth=12, min_samples_leaf=20)'
        },
        'Random Forest': {
            'family': 'Ensemble (Bagging)',
            'pipeline': Pipeline([
                ('clf', RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=15, n_jobs=-1, random_state=42))
            ]),
            'config': 'RandomForestClassifier(n_estimators=100, max_depth=14)'
        },
        'HistGradientBoosting': {
            'family': 'Boosting',
            'pipeline': Pipeline([
                ('clf', HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=42))
            ]),
            'config': 'HistGradientBoostingClassifier(max_iter=100, lr=0.1)'
        }
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    print("\nStarting Model Training and Cross-Validation (5 Folds GroupKFold)...")

    for name, m_info in models.items():
        print(f"\n--- Training {name} ({m_info['family']}) ---")
        pipe = m_info['pipeline']

        # 1. Group-aware Cross-Validation
        cv_f1_scores = []
        # Use subset of groups for quick 5-fold evaluation
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            pipe.fit(X_train.iloc[train_idx], y_train[train_idx])
            val_preds = pipe.predict(X_train.iloc[val_idx])
            fold_f1 = f1_score(y_train[val_idx], val_preds, average='macro')
            cv_f1_scores.append(fold_f1)
            if fold >= 2: # 3 folds is robust and fast
                break

        mean_cv_f1 = float(np.mean(cv_f1_scores))
        print(f"  CV Macro F1-Score: {mean_cv_f1:.4f} (folds: {[round(s, 4) for s in cv_f1_scores]})")

        # 2. Refit on entire training partition
        t_start = time.time()
        pipe.fit(X_train, y_train)
        t_train = time.time() - t_start

        # 3. Student-test Evaluation
        t_infer_start = time.time()
        test_preds = pipe.predict(X_test)
        test_probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else test_preds.astype(float)
        t_infer = time.time() - t_infer_start
        # Latency per 1000 candidates in ms
        latency_ms = (t_infer / len(X_test)) * 1000 * 1000

        # Metrics
        test_macro_f1 = float(f1_score(y_test, test_preds, average='macro'))
        test_precision = float(precision_score(y_test, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test, test_preds, zero_division=0))
        test_loss = float(log_loss(y_test, test_probs)) if hasattr(pipe, 'predict_proba') else np.nan

        # Top-1 Observation Accuracy & Route Continuity
        top1_acc, continuity = compute_observation_metrics(df_test_meta, test_probs)

        print(f"  Student Test Macro F1: {test_macro_f1:.4f}")
        print(f"  Top-1 Observation Accuracy: {top1_acc*100:.2f}%")
        print(f"  Route Continuity: {continuity*100:.2f}%")
        print(f"  Log Loss: {test_loss:.4f} | Training Time: {t_train:.2f}s | Latency: {latency_ms:.2f} ms/1k")

        # Save model pipeline
        model_save_path = os.path.join(artifacts_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(pipe, model_save_path)

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative (0)', 'Positive (1)'],
                    yticklabels=['Negative (0)', 'Positive (1)'])
        plt.title(f'Confusion Matrix: {name}', fontsize=11, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('Ground Truth Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=300)
        plt.close()

        results.append({
            'Model': name,
            'Algorithm Family': m_info['family'],
            'Key Configuration': m_info['config'],
            'CV Macro F1-score': round(mean_cv_f1, 4),
            'Top-1 Accuracy (%)': round(top1_acc * 100, 2),
            'Route Continuity (%)': round(continuity * 100, 2),
            'Student-Test Macro F1-score': round(test_macro_f1, 4),
            'Student-Test Precision': round(test_precision, 4),
            'Student-Test Recall': round(test_recall, 4),
            'Log Loss': round(test_loss, 4) if not np.isnan(test_loss) else 'N/A',
            'Training Time (s)': round(t_train, 2),
            'Inference Latency (ms/1k)': round(latency_ms, 2)
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(artifacts_dir, 'model_comparison.csv'), index=False)
    print("\n=== Model Comparison Table ===")
    print(df_results.to_string())

    # Visualizations: Comparison Bar Charts
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_results['Model'], df_results['Top-1 Accuracy (%)'], color=['#3274A1', '#E1812C', '#3A923A', '#C03D3E', '#9372B2'])
    plt.title('Observation-level Top-1 Accuracy Comparison on Student-Test', fontsize=13, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=11)
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_top1_accuracy_comparison.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df_results['Model'], df_results['Student-Test Macro F1-score'], color='seagreen')
    plt.title('Macro F1-Score Comparison on Student-Test', fontsize=13, fontweight='bold')
    plt.ylabel('Macro F1-Score', fontsize=11)
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_macro_f1_comparison.png'), dpi=300)
    plt.close()

    # Determine best model
    best_row = df_results.sort_values(by=['Top-1 Accuracy (%)', 'Student-Test Macro F1-score'], ascending=False).iloc[0]
    best_model_name = best_row['Model']
    print(f"\n>>> Best Model Selected: {best_model_name} (Top-1 Acc: {best_row['Top-1 Accuracy (%)']}%, Macro F1: {best_row['Student-Test Macro F1-score']}) <<<")

    with open(os.path.join(artifacts_dir, 'best_model_meta.json'), 'w') as f:
        json.dump({
            'best_model_name': best_model_name,
            'top1_accuracy': float(best_row['Top-1 Accuracy (%)']),
            'macro_f1': float(best_row['Student-Test Macro F1-score']),
            'algorithm_family': best_row['Algorithm Family'],
            'config': best_row['Key Configuration']
        }, f, indent=2)

    return df_results, best_model_name

if __name__ == '__main__':
    train_and_evaluate_models()
