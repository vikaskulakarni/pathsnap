"""
PathSnap Master Pipeline Runner
Executes:
1. EDA and dataset validation
2. Vectorized feature extraction and candidate generation
3. 5-model training, cross-validation, and comparative evaluation
4. Bayesian hyperparameter optimization (Optuna TPE)
5. HMM Viterbi route reconstruction and continuity analysis
6. Interactive Folium map generation
"""

import sys
import os
import time

sys.path.append('src')

from eda import run_eda
from preprocessing import build_dataset
from models import train_and_evaluate_models
from tuning import run_bayesian_optimization
from hmm_decoder import evaluate_hmm_reconstruction
from visualizations import generate_all_evaluation_maps

def main():
    print("=" * 65)
    print("   PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION   ")
    print("=" * 65)

    # Stage 1: EDA
    print("\n>>> STAGE 1: Exploratory Data Analysis & Statistics <<<")
    run_eda()

    # Stage 2: Preprocessing & Features
    print("\n>>> STAGE 2: Spatial Indexing & Feature Engineering <<<")
    build_dataset(n_train_trips=25)

    # Stage 3: 5 Model Training & Comparison
    print("\n>>> STAGE 3: Multi-Family Model Training & Evaluation <<<")
    train_and_evaluate_models()

    # Stage 4: Bayesian Tuning
    print("\n>>> STAGE 4: Bayesian Hyperparameter Optimization <<<")
    run_bayesian_optimization(n_trials=25)

    # Stage 5: HMM Viterbi Reconstruction
    print("\n>>> STAGE 5: HMM Viterbi Sequential Route Reconstruction <<<")
    evaluate_hmm_reconstruction()

    # Stage 6: Folium Visualization Maps
    print("\n>>> STAGE 6: Generating Interactive Folium Evaluation Maps <<<")
    generate_all_evaluation_maps()

    print("\n" + "=" * 65)
    print("   ALL STAGES COMPLETED SUCCESSFULLY BEFORE 11:30 AM!   ")
    print("=" * 65)

if __name__ == '__main__':
    main()
