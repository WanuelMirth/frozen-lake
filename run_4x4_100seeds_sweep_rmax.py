# run_4x4_100seeds_sweep_rmax.py
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import random
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from agents.rmax_agent import RMaxAgent

# Configurations to evaluate (100 seeds, 5,000 episodes, 100 steps limit)
CONFIGS = {
    "RMax_Best_Config": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "total_episodes": 5000,
        "max_steps_per_episode": 100,              # Explicit 100 steps limit for 4x4
        "discount_rate": 0.9849838462915839,       
        "m": 14, 
        "render": False,
        "desc": "R-Max Best"
    }
}

NUM_SEEDS = 100
BASE_DIR = "results/4x4_sweep_100seeds"

def train_single_seed(config_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    config = CONFIGS[config_name]
    run_dir = os.path.join(BASE_DIR, config_name, f"RMax_seed{seed}")
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
        
        agent = RMaxAgent(env.observation_space, env.action_space, config)
        
        rewards = []
        for episode in range(1, config["total_episodes"] + 1):
            current_seed = seed + episode
            state, info = env.reset(seed=current_seed)
            
            terminated, truncated = False, False
            episode_reward = 0
            
            for step in range(config["max_steps_per_episode"]):
                action = agent.choose_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.learn(state, action, reward, new_state, terminated)
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

def generate_rmax_plot(csv_path, window_size=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV {csv_path} is empty.")
            return None

        # Smoothing
        df['smoothed_mean'] = df['mean'].rolling(window=window_size, min_periods=1).mean()

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        final_perf = df['smoothed_mean'].iloc[-1000:].mean()
        
        color = 'crimson'  # High-contrast crimson color matching fast convergence runs
        label = f'Success Rate (Avg Last 1000 Ep: {final_perf:.4f})'
        
        # Plot mean line
        ax.plot(df['episode'], df['smoothed_mean'], color=color, linewidth=2, label=label)

        title_label = "R-Max (Best Configuration - 100 Seeds)"
        ax.set_title(f"Learning Curve: {title_label}", fontsize=14, pad=15)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Reward (Smoothed)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)

        # Mark the convergence episode on x axis (70% threshold for 4x4 grid)
        threshold = 0.70
        cross_indices = df[df['smoothed_mean'] >= threshold].index
        conv_ep = int(df.loc[cross_indices[0], 'episode']) if len(cross_indices) > 0 else None
        
        if conv_ep is not None:
            ax.axvline(x=conv_ep, color='dimgray', linestyle=':', alpha=0.8, label=f'Convergence (70%): Ep {conv_ep}')
            
            # Prevent overlapping numbers by custom xticks
            max_ep = df['episode'].max()
            current_ticks = [0, 1000, 2000, 3000, 4000, 5000]
            tick_spacing = 50
            new_ticks = [t for t in current_ticks if abs(t - conv_ep) > tick_spacing]
            new_ticks.append(conv_ep)
            ax.set_xticks(sorted(new_ticks))

        ax.legend(loc='lower right', frameon=True)
        fig.tight_layout()

        plot_path = csv_path.replace('.csv', '_smooth_plot.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"R-Max custom plot saved to: {plot_path}")
        
        # Copy to results directory for convenient access
        results_dir = "results"
        filename = os.path.basename(plot_path)
        dest = os.path.join(results_dir, filename)
        import shutil
        shutil.copy(plot_path, dest)
        print(f"Copied plot for easy access to: {dest}")
        
        plt.close(fig)
        return df

    except Exception as e:
        print(f"An error occurred plotting {csv_path}: {e}")
        return None

def run_experiment():
    print(f"===== STARTING 4x4 SLIPPERY SWEEP FOR R-MAX (100 SEEDS, 5,000 EPISODES, 100 STEPS) =====")
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

    # Plotting separate learning curves with custom R-Max plotting
    for name in CONFIGS:
        summary_path = os.path.join(BASE_DIR, f"AGGREGATED_{name}.csv")
        if name in aggregated_dfs:
            df = aggregated_dfs[name]
            df['smoothed_mean'] = df['mean'].rolling(window=100, min_periods=1).mean()
            final_perf = df['smoothed_mean'].iloc[-1000:].mean()
            cross_indices = df[df['smoothed_mean'] >= 0.70].index
            first_70_ep = int(df.loc[cross_indices[0], 'episode']) if len(cross_indices) > 0 else None
            print(f"[{name}] Convergence Episode (rolling 100-ep Avg >= 70%): {first_70_ep} | Final 1000 Ep Avg Success: {final_perf*100:.2f}%")
            
        print(f"Generating separate smooth plot for {name} using R-Max custom layout...")
        generate_rmax_plot(summary_path, window_size=100)
        
    print("All tasks finished successfully.")

if __name__ == "__main__":
    run_experiment()
