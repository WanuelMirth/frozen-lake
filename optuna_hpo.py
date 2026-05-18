import optuna
import main
import os
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
STUDY_NAME = "UCT_RBQL_FrozenLake_Optimization"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
N_TRIALS = 100
N_JOBS = -1  # Use all available CPU cores

BASE_CONFIG = {
    "env_name": "FrozenLake-v1",
    "is_slippery": True,
    "map_name": "4x4",
    "agent": "UCT-RBQL",
    "total_episodes": 2000,
    "max_steps_per_episode": 200,
    "render": False,
    "seed": 42
}

def objective(trial):
    # 1. Define Search Space
    exploration_constant_c = trial.suggest_float("exploration_constant_c", 0.01, 1.0, log=True)
    discount_rate = trial.suggest_float("discount_rate", 0.9, 0.999)
    
    # 2. Setup Config
    config = BASE_CONFIG.copy()
    config.update({
        "exploration_constant_c": exploration_constant_c,
        "discount_rate": discount_rate,
    })
    
    # 3. Run Training
    timestamp = datetime.now().strftime("%H%M%S_%f")
    run_name = f"Optuna_{trial.number}_{timestamp}"
    
    results_dir, duration = main.train(run_name, config)
    
    # 4. Extract Metrics
    try:
        metrics_path = os.path.join(results_dir, "metrics.csv")
        df = pd.read_csv(metrics_path)
        
        if df.empty:
            return 0.0
            
        # We want to maximize the average reward of the last 100 episodes
        final_performance = df['reward'].tail(100).mean()
        
        # Optional: Penalize long episodes or reward fast convergence
        # But for FrozenLake, successful reward is the main goal.
        
        return final_performance
    except Exception as e:
        print(f"Error extracting metrics for trial {trial.number}: {e}")
        return 0.0

if __name__ == "__main__":
    # Check if optuna is installed (just in case)
    try:
        import optuna
    except ImportError:
        print("Optuna not found. Please install it using: pip install optuna")
        exit(1)

    print(f"Starting Optuna Study: {STUDY_NAME}")
    print(f"Using {N_JOBS} parallel jobs.")

    # Create study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="maximize",
        load_if_exists=True
    )

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    print("\n--- Optimization Finished ---")
    print(f"Best trial: {study.best_trial.number}")
    print(f"  Value: {study.best_value}")
    print(f"  Params: {study.best_params}")

    # --- ADDITIONAL SUMMARY ---
    df = study.trials_dataframe()
    # Sort by value (descending since we maximize)
    df = df.sort_values(by="value", ascending=False)
    
    print("\n--- Top 5 Trials ---")
    print(df[['number', 'value', 'params_exploration_constant_c', 'params_discount_rate']].head(5).to_string(index=False))

    # Save all trials to CSV for deeper analysis
    csv_filename = f"optuna_results_{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nAll trial results saved to: {csv_filename}")
    
    print("\nTip: Use 'optuna-dashboard " + STORAGE_NAME + "' to visualize these results in your browser!")
