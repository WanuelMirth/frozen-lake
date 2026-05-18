import optuna
import main
import os
import pandas as pd
import numpy as np
from datetime import datetime
import random

# --- CONFIGURATION ---
STUDY_NAME = "UCT_RBQL_Multi_Objective"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"
OLD_STUDY_NAME = "UCT_RBQL_FrozenLake_Optimization"
OLD_STORAGE_NAME = f"sqlite:///{OLD_STUDY_NAME}.db"

N_TRIALS = 100
N_JOBS = -1
EVAL_SEEDS = [111, 222, 333]
CONVERGENCE_THRESHOLD = 0.7000

BASE_CONFIG = {
    "env_name": "FrozenLake-v1",
    "is_slippery": True,
    "map_name": "4x4",
    "agent": "UCT-RBQL",
    "total_episodes": 2000,
    "max_steps_per_episode": 200,
    "render": False,
}

def calculate_convergence_speed(rewards, threshold):
    rolling_avg = pd.Series(rewards).rolling(window=100).mean().fillna(0).values
    
    # Find the FIRST episode where the rolling average reaches the threshold
    above_threshold_indices = np.where(rolling_avg >= threshold)[0]
    
    if len(above_threshold_indices) == 0:
        return len(rewards) # Failed to ever reach threshold
        
    return above_threshold_indices[0] + 1

def objective(trial):
    # 1. Define Search Space
    exploration_constant_c = trial.suggest_float("exploration_constant_c", 0.01, 1.0, log=True)
    discount_rate = trial.suggest_float("discount_rate", 0.9, 0.999)
    
    seed_performances = []
    seed_convergences = []
    
    for seed in EVAL_SEEDS:
        # 2. Setup Config
        config = BASE_CONFIG.copy()
        config.update({
            "exploration_constant_c": exploration_constant_c,
            "discount_rate": discount_rate,
            "seed": seed
        })
        
        # 3. Run Training
        timestamp = datetime.now().strftime("%H%M%S_%f")
        run_name = f"Trial_{trial.number}_Seed_{seed}_{timestamp}"
        
        results_dir, duration = main.train(run_name, config, trial=trial)
        
        # 4. Extract Metrics
        try:
            metrics_path = os.path.join(results_dir, "metrics.csv")
            df = pd.read_csv(metrics_path)
            
            if df.empty:
                seed_performances.append(0.0)
                seed_convergences.append(BASE_CONFIG["total_episodes"])
                continue
                
            # Objective 0: Mean reward of last 500 episodes
            final_perf = df['reward'].tail(500).mean()
            seed_performances.append(final_perf)
            
            # Objective 1: Convergence speed
            conv_speed = calculate_convergence_speed(df['reward'].values, CONVERGENCE_THRESHOLD)
            seed_convergences.append(conv_speed)
            
        except Exception as e:
            print(f"Error extracting metrics for trial {trial.number}, seed {seed}: {e}")
            seed_performances.append(0.0)
            seed_convergences.append(BASE_CONFIG["total_episodes"])

    mean_performance = round(np.mean(seed_performances), 4)
    mean_convergence = round(np.mean(seed_convergences), 4)
    
    return mean_performance, mean_convergence

def bootstrap_study(study):
    if not os.path.exists(f"{OLD_STUDY_NAME}.db"):
        print(f"Old database {OLD_STUDY_NAME}.db not found. Skipping bootstrap.")
        return

    try:
        old_study = optuna.load_study(study_name=OLD_STUDY_NAME, storage=OLD_STORAGE_NAME)
        completed_trials = [t for t in old_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            return
            
        # Sort by value (maximize)
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        top_trials = completed_trials[:3]
        print(f"Bootstrapping with {len(top_trials)} trials from Phase 1...")
        
        for t in top_trials:
            study.enqueue_trial(t.params)
            print(f"  Enqueued: {t.params}")
            
    except Exception as e:
        print(f"Error during bootstrapping: {e}")

if __name__ == "__main__":
    print(f"Starting Multi-Objective Optuna Study: {STUDY_NAME}")
    
    # Use TPESampler for multi-objective optimization
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        directions=["maximize", "minimize"],
        load_if_exists=True
    )

    # Bootstrap from old study if it's the first time
    if len(study.trials) == 0:
        bootstrap_study(study)

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    print("\n--- Optimization Finished ---")
    
    # Pareto Front
    print("\n--- Pareto Front (Best Trade-offs) ---")
    for trial in study.best_trials:
        print(f"Trial {trial.number}: Performance={trial.values[0]:.4f}, Convergence={trial.values[1]:.1f}")
        print(f"  Params: {trial.params}")

    # Save results
    df = study.trials_dataframe()
    csv_filename = f"optuna_results_{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
