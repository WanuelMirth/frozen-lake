# seed_sweep.py
import os
import pandas as pd
from datetime import datetime
import json
import numpy as np
import main # Import train function
from concurrent.futures import ProcessPoolExecutor, as_completed

CHAMPION_CONFIGS = {
    "UCT_RBQL_Deep_Pareto_Trial74": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "agent": "UCT-RBQL-Deep",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9926714709783417,
        "exploration_constant_c": 0.059991060225751965,
        "render": False,
    },
    "UCT_RBQL_Deep_Pareto_Trial92": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "agent": "UCT-RBQL-Deep",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9843205299768125,
        "exploration_constant_c": 0.09900852192118169,
        "render": False,
    }
}

# --- 2. DEFINE THE SWEEP PARAMETERS ---
NUM_SEEDS = 25 # Number of runs per configuration

def run_single_seed(args):
    config_name, seed, base_config = args
    config = base_config.copy()
    config['seed'] = seed # Add seed
    
    # Unique timestamp with microseconds to avoid name collisions
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_name = f"{config_name}_seed{seed}_{timestamp}"
    
    print(f"Starting seed {seed} for {config_name}...")
    results_path, _ = main.train(run_name, config)
    return results_path

# --- 3. FUNCTION TO PERFORM THE SEED SWEEP ---
def run_seed_sweep(config_name):
    try:
        base_config = CHAMPION_CONFIGS[config_name]
    except KeyError:
        print(f"Error: Champion configuration '{config_name}' not found. Available: {list(CHAMPION_CONFIGS.keys())}")
        return []
        
    print(f"\n===== STARTING PARALLEL SEED SWEEP FOR: {config_name} ({NUM_SEEDS} SEEDS) =====")
    
    tasks = [(config_name, seed, base_config) for seed in range(NUM_SEEDS)]
    all_run_dirs = [None] * NUM_SEEDS
    
    # Limit max workers based on CPU cores
    max_workers = min(NUM_SEEDS, os.cpu_count() or 4)
    print(f"Running sweep using {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_seed, task): task[1] for task in tasks}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                results_path = future.result()
                all_run_dirs[seed] = results_path
            except Exception as e:
                print(f"Error running seed {seed}: {e}")
                
    return [r for r in all_run_dirs if r is not None]

# --- 4. FUNCTION FOR AGGREGATION ---
def aggregate_results(all_dirs, base_run_name):
    all_dfs = []
    for run_dir in all_dirs:
        try:
            df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: metrics.csv not found in {run_dir}")

    if not all_dfs:
        print("No data found to aggregate.")
        return

    combined_df = pd.concat(all_dfs)
    aggregated_df = combined_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    aggregated_df['std'] = aggregated_df['std'].fillna(0)
    
    summary_path = os.path.join("results", f"AGGREGATED_{base_run_name}.csv")
    aggregated_df.to_csv(summary_path, index=False)
    print(f"\nAggregated results saved under: {summary_path}")

if __name__ == "__main__":
    configs_to_run = [
        "UCT_RBQL_Deep_Pareto_Trial74",
        "UCT_RBQL_Deep_Pareto_Trial92"
    ]
    
    for config_name in configs_to_run:
        run_directories = run_seed_sweep(config_name)
        if run_directories:
            aggregate_results(run_directories, config_name)