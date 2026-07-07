# optuna_rmax.py
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import main_rmax
import os
import pandas as pd
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
STUDY_NAME = "RMax_Multi_Objective_4x4"
JOURNAL_PATH = f"./{STUDY_NAME}.log"

N_TRIALS = 100
N_JOBS = -1

# The exact 10 independent seeds used in the QUEST HPO for 4x4
EVAL_SEEDS = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
CONVERGENCE_THRESHOLD = 0.7000

BASE_CONFIG = {
    "env_name": "FrozenLake-v1",
    "is_slippery": True,
    "map_name": "4x4",
    "total_episodes": 2000,
    "max_steps_per_episode": 100,
    "render": False,
}

def calculate_convergence_speed(rewards, threshold):
    rolling_avg = pd.Series(rewards).rolling(window=100).mean().fillna(0).values
    above_threshold_indices = np.where(rolling_avg >= threshold)[0]
    
    if len(above_threshold_indices) == 0:
        return len(rewards)
        
    return above_threshold_indices[0] + 1

def objective(trial):
    # 1. Suggest Hyperparameters
    discount_rate = trial.suggest_float("discount_rate", 0.9, 0.999)
    m = trial.suggest_int("m", 1, 50)
    
    seed_performances = []
    seed_convergences = []
    
    # Seeds are evaluated sequentially within each trial
    for seed in EVAL_SEEDS:
        config = BASE_CONFIG.copy()
        config.update({
            "discount_rate": discount_rate,
            "m": m,
            "seed": seed
        })
        
        timestamp = datetime.now().strftime("%H%M%S_%f")
        run_name = f"optuna_rmax/Trial_{trial.number}_Seed_{seed}_{timestamp}"
        
        # Run training loop
        results_dir, duration = main_rmax.train(run_name, config, trial=trial)
        
        try:
            metrics_path = os.path.join(results_dir, "metrics.csv")
            df = pd.read_csv(metrics_path)
            
            if df.empty:
                seed_performances.append(0.0)
                seed_convergences.append(BASE_CONFIG["total_episodes"])
                continue
                
            # Mean success rate of last 500 episodes
            final_perf = df['reward'].tail(500).mean()
            seed_performances.append(final_perf)
            
            # Episode where rolling 100-episode average reward >= 70%
            conv_speed = calculate_convergence_speed(df['reward'].values, CONVERGENCE_THRESHOLD)
            seed_convergences.append(conv_speed)
            
        except Exception as e:
            print(f"Error extracting metrics for trial {trial.number}, seed {seed}: {e}")
            seed_performances.append(0.0)
            seed_convergences.append(BASE_CONFIG["total_episodes"])

    mean_performance = round(np.mean(seed_performances), 4)
    mean_convergence = round(np.mean(seed_convergences), 4)
    
    return mean_performance, mean_convergence

if __name__ == "__main__":
    print(f"Starting Reproducible Multi-Objective Optuna Study for R-Max: {STUDY_NAME}")
    
    # Initialize concurrent storage engine
    storage = JournalStorage(JournalFileStorage(JOURNAL_PATH))
    
    # Seeding the Optuna sampler for HPO reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)
    
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True
    )

    # Parallel trials execution
    print(f"Launching {N_TRIALS} trials across {N_JOBS} jobs via JournalStorage...")
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)

    print("\n--- Optimization Finished ---")
    
    print("\n--- Pareto Front (Best Trade-offs) ---")
    for trial in study.best_trials:
        if trial.values:
            print(f"Trial {trial.number}: Performance={trial.values[0]:.4f}, Convergence={trial.values[1]:.1f}")
            print(f"  Params: {trial.params}")

    df = study.trials_dataframe()
    csv_filename = f"optuna_results_{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
