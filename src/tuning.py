import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from models import compute_observation_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_bayesian_optimization(n_trials=25, plots_dir='plots', artifacts_dir='artifacts'):
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading data for Bayesian Hyperparameter Tuning...")
    X_train = joblib.load('data/X_train.joblib')
    y_train = joblib.load('data/y_train.joblib')
    groups_train = joblib.load('data/groups_train.joblib')
    X_test = joblib.load('data/X_test.joblib')
    y_test = joblib.load('data/y_test.joblib')
    df_test_meta = joblib.load('data/test_meta.joblib')

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_train, y_train, groups=groups_train))[:3] # 3 folds for fast reliable tuning

    trial_records = []

    def objective(trial):
        # 1. Define hyperparameter search space
        learning_rate = trial.suggest_float('learning_rate', 0.02, 0.25, log=True)
        max_iter = trial.suggest_int('max_iter', 60, 200, step=20)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 15, 63)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 10, 50)
        l2_regularization = trial.suggest_float('l2_regularization', 1e-4, 10.0, log=True)

        clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=42
        )

        fold_scores = []
        for train_idx, val_idx in splits:
            clf.fit(X_train.iloc[train_idx], y_train[train_idx])
            val_preds = clf.predict(X_train.iloc[val_idx])
            fold_scores.append(f1_score(y_train[val_idx], val_preds, average='macro'))

        mean_score = float(np.mean(fold_scores))

        trial_records.append({
            'trial': trial.number,
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_leaf': min_samples_leaf,
            'l2_regularization': l2_regularization,
            'cv_macro_f1': mean_score
        })

        return mean_score

    print(f"\nRunning Optuna TPE Bayesian Search ({n_trials} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    t_opt = time.time() - t0

    print(f"Optimization finished in {t_opt:.2f}s!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best CV Macro F1: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")

    df_trials = pd.DataFrame(trial_records)
    df_trials.to_csv(os.path.join(artifacts_dir, 'bayesian_tuning_trials.csv'), index=False)

    # Plot Optimization History
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_trials['trial'], df_trials['cv_macro_f1'], marker='o', color='teal', alpha=0.6, label='Trial Score')
    plt.plot(df_trials['trial'], df_trials['cv_macro_f1'].cummax(), color='crimson', linewidth=2, label='Current Best')
    plt.title('Bayesian Optimization History (Optuna TPE)', fontsize=13, fontweight='bold')
    plt.xlabel('Trial Number', fontsize=11)
    plt.ylabel('Validation Macro F1-score', fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bayesian_opt_history.png'), dpi=300)
    plt.close()

    # Refit Untuned vs Tuned Models for side-by-side comparison
    print("\nRefitting untuned model...")
    untuned_clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=31, learning_rate=0.1, random_state=42)
    untuned_clf.fit(X_train, y_train)

    t_start = time.time()
    untuned_preds = untuned_clf.predict(X_test)
    untuned_probs = untuned_clf.predict_proba(X_test)[:, 1]
    untuned_latency = ((time.time() - t_start) / len(X_test)) * 1e6
    untuned_top1, untuned_cont = compute_observation_metrics(df_test_meta, untuned_probs)
    untuned_f1 = f1_score(y_test, untuned_preds, average='macro')
    untuned_loss = log_loss(y_test, untuned_probs)

    print("Refitting tuned model with best hyperparameters...")
    best_p = study.best_params
    tuned_clf = HistGradientBoostingClassifier(
        learning_rate=best_p['learning_rate'],
        max_iter=best_p['max_iter'],
        max_leaf_nodes=best_p['max_leaf_nodes'],
        min_samples_leaf=best_p['min_samples_leaf'],
        l2_regularization=best_p['l2_regularization'],
        random_state=42
    )
    tuned_clf.fit(X_train, y_train)

    t_start = time.time()
    tuned_preds = tuned_clf.predict(X_test)
    tuned_probs = tuned_clf.predict_proba(X_test)[:, 1]
    tuned_latency = ((time.time() - t_start) / len(X_test)) * 1e6
    tuned_top1, tuned_cont = compute_observation_metrics(df_test_meta, tuned_probs)
    tuned_f1 = f1_score(y_test, tuned_preds, average='macro')
    tuned_loss = log_loss(y_test, tuned_probs)

    # Save tuned model
    joblib.dump(tuned_clf, os.path.join(artifacts_dir, 'final_tuned_model.joblib'))

    comparison_data = [
        {
            'Configuration': 'Untuned Baseline',
            'CV Macro F1': 0.7463,
            'Student-Test Top-1 Accuracy (%)': round(untuned_top1 * 100, 2),
            'Student-Test Macro F1': round(untuned_f1, 4),
            'Route Continuity (%)': round(untuned_cont * 100, 2),
            'Log Loss': round(untuned_loss, 4),
            'Inference Latency (ms/1k)': round(untuned_latency, 2)
        },
        {
            'Configuration': 'Tuned (Bayesian Opt)',
            'CV Macro F1': round(study.best_value, 4),
            'Student-Test Top-1 Accuracy (%)': round(tuned_top1 * 100, 2),
            'Student-Test Macro F1': round(tuned_f1, 4),
            'Route Continuity (%)': round(tuned_cont * 100, 2),
            'Log Loss': round(tuned_loss, 4),
            'Inference Latency (ms/1k)': round(tuned_latency, 2)
        }
    ]

    df_comp = pd.DataFrame(comparison_data)
    df_comp.to_csv(os.path.join(artifacts_dir, 'tuning_comparison.csv'), index=False)
    print("\n=== Untuned vs Tuned Model Performance Comparison ===")
    print(df_comp.to_string())

    with open(os.path.join(artifacts_dir, 'best_hyperparameters.json'), 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_cv_f1': float(study.best_value),
            'opt_time_seconds': float(t_opt),
            'n_trials': n_trials
        }, f, indent=2)

    return df_comp, study.best_params

if __name__ == '__main__':
    run_bayesian_optimization()
