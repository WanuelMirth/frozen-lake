# run_4x4_100seeds_sweep.py
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import random
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from agents.quest_agent import QUESTAgent
from plot import generate_individual_plot

# Configurations to evaluate (100 seeds, 5,000 episodes, 100 steps limit)
CONFIGS = {
    "QUEST_Pareto_Trial40": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "total_episodes": 5000,
        "max_steps_per_episode": 100,              # Explicit 100 steps limit for 4x4
        "discount_rate": 0.9889047072670322,       
        "exploration_constant_c": 0.08240426095596366, 
        "render": False,
        "desc": "Trial 40"
    }
}

NUM_SEEDS = 100
BASE_DIR = "results/4x4_sweep_100seeds"

def train_single_seed(config_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    config = CONFIGS[config_name]
    run_dir = os.path.join(BASE_DIR, config_name, f"QUEST_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
        
    csv_path = f"{run_dir}/metrics.csv"
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['episode', 'steps', 'reward'])
        
        env = gym.make(
            config["env_name"], 
            is_slippery=config["is_slippery"], 
            map_name=config["map_name"],
            max_episode_steps=config["max_steps_per_episode"]
        )
        
        agent = QUESTAgent(env.observation_space, env.action_space, config)
        
        rewards = []
        for episode in range(1, config["total_episodes"] + 1):
            current_seed = seed + episode
            state, info = env.reset(seed=current_seed)
            
            terminated, truncated = False, False
            episode_reward = 0
            
            for step in range(config["max_steps_per_episode"]):
                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.learn(state, action, reward, new_state)
                episode_reward += reward
                state = new_state
                if terminated or truncated:
                    break
                    
            agent.on_episode_end(episode, episode_reward)
            csv_writer.writerow([episode, step + 1, episode_reward])
            rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                recent_avg = np.mean(rewards[-100:]) * 100
                print(f"[{config_name}] Seed {seed} | Episode {episode}/{config['total_episodes']} | Recent Success (last 100): {recent_avg:.1f}%", flush=True)
            
        env.close()
    
    print(f"Finished {config_name} - Seed {seed} evaluation.")
    return run_dir

def run_experiment():
    print(f"===== STARTING 4x4 SLIPPERY SWEEP (100 SEEDS, 5,000 EPISODES, 100 STEPS) =====")
    start_time = time.time()
    
    os.makedirs(BASE_DIR, exist_ok=True)
    
    tasks = []
    for config_name in CONFIGS:
        for seed in range(NUM_SEEDS):
            tasks.append((config_name, seed))
            
    # Run in parallel using ProcessPoolExecutor
    max_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Running sweep using {max_workers} parallel workers...")
    
    run_directories = {name: [None]*NUM_SEEDS for name in CONFIGS}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_single_seed, name, seed): (name, seed) for name, seed in tasks}
        for future in as_completed(futures):
            name, seed = futures[future]
            try:
                results_dir = future.result()
                run_directories[name][seed] = results_dir
            except Exception as e:
                print(f"Error running {name} seed {seed}: {e}")
                
    elapsed = time.time() - start_time
    print(f"Sweep completed in {elapsed:.1f} seconds.")
    
    # Aggregate and Save results
    aggregated_dfs = {}
    
    for name in CONFIGS:
        print(f"\nAggregating results for {name}...")
        all_dfs = []
        for run_dir in run_directories[name]:
            if run_dir and os.path.exists(os.path.join(run_dir, "metrics.csv")):
                df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
                all_dfs.append(df)
                
        if not all_dfs:
            print(f"Error: No metrics files found for {name}.")
            continue
            
        combined_df = pd.concat(all_dfs)
        aggregated_df = combined_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
        aggregated_df['std'] = aggregated_df['std'].fillna(0)
        
        summary_path = os.path.join(BASE_DIR, f"AGGREGATED_{name}.csv")
        aggregated_df.to_csv(summary_path, index=False)
        print(f"Aggregated data saved to: {summary_path}")
        
        # Save JSON
        json_path = os.path.join(BASE_DIR, f"{name}_results.json")
        seeds_data = {}
        for seed in range(NUM_SEEDS):
            df_seed = pd.read_csv(os.path.join(run_directories[name][seed], "metrics.csv"))
            seeds_data[str(seed)] = {
                "rewards": df_seed["reward"].tolist(),
                "steps": df_seed["steps"].tolist()
            }
            
        json_output = {
            "config": CONFIGS[name],
            "mean_rewards": aggregated_df["mean"].tolist(),
            "std_rewards": aggregated_df["std"].tolist(),
            "seeds": seeds_data
        }
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
            
        aggregated_dfs[name] = aggregated_df

    # Plotting separate learning curves exactly as the aggregated pareto plots
    for name in CONFIGS:
        summary_path = os.path.join(BASE_DIR, f"AGGREGATED_{name}.csv")
        if name in aggregated_dfs:
            df = aggregated_dfs[name]
            df['smoothed_mean'] = df['mean'].rolling(window=100, min_periods=1).mean()
            final_perf = df['smoothed_mean'].iloc[-1000:].mean()
            cross_indices = df[df['smoothed_mean'] >= 0.70].index
            first_70_ep = int(df.loc[cross_indices[0], 'episode']) if len(cross_indices) > 0 else None
            print(f"[{name}] Convergence Episode (rolling 100-ep Avg >= 70%): {first_70_ep} | Final 1000 Ep Avg Success: {final_perf*100:.2f}%")
            
        print(f"Generating separate smooth plot for {name}...")
        generate_individual_plot(summary_path, window_size=100)
        
    print("All tasks finished successfully.")

if __name__ == "__main__":
    run_experiment()
