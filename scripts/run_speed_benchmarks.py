# run_speed_benchmarks.py
import os
import time
import gymnasium as gym
import numpy as np
import pandas as pd
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from agents.quest_agent import QUESTAgent
from agents.rmax_agent import RMaxAgent

# Configurations
CONFIGS = {
    "QUEST_4x4": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "total_episodes": 5000,
        "max_steps_per_episode": 100,
        "discount_rate": 0.9889047072670322,
        "exploration_constant_c": 0.08240426095596366,
        "max_iterations_multiplier": 5,
        "agent_type": "QUEST"
    },
    "RMax_4x4": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "total_episodes": 5000,
        "max_steps_per_episode": 100,
        "discount_rate": 0.9849838462915839,
        "m": 14,
        "agent_type": "RMax"
    },
    "QUEST_8x8": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "8x8",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9988452099891397,
        "exploration_constant_c": 0.01380488848357483,
        "max_iterations_multiplier": 5,
        "agent_type": "QUEST"
    },
    "RMax_8x8": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "8x8",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9989859250982733,
        "m": 8,
        "agent_type": "RMax"
    }
}

NUM_SEEDS = 10
SEEDS = list(range(NUM_SEEDS))

def run_single_seed(config_name, seed):
    config = CONFIGS[config_name]
    env = gym.make(
        config["env_name"],
        is_slippery=config["is_slippery"],
        map_name=config["map_name"],
        max_episode_steps=config["max_steps_per_episode"]
    )
    
    # Initialize the correct agent
    if config["agent_type"] == "QUEST":
        agent = QUESTAgent(env.observation_space, env.action_space, config)
    else:
        agent = RMaxAgent(env.observation_space, env.action_space, config)
        
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    episode_times = []
    cumulative_times = []
    rewards = []
    
    elapsed_total = 0.0
    
    for episode in range(1, config["total_episodes"] + 1):
        current_seed = seed + episode
        state, info = env.reset(seed=current_seed)
        
        terminated, truncated = False, False
        episode_reward = 0
        
        # Measure duration of this episode
        start_time = time.time()
        for step in range(config["max_steps_per_episode"]):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            
            if config["agent_type"] == "QUEST":
                agent.learn(state, action, reward, new_state)
            else:
                agent.learn(state, action, reward, new_state, terminated)
                
            episode_reward += reward
            state = new_state
            if terminated or truncated:
                break
                
        agent.on_episode_end(episode, episode_reward)
        duration = time.time() - start_time
        
        elapsed_total += duration
        
        episode_times.append(duration)
        cumulative_times.append(elapsed_total)
        rewards.append(episode_reward)
        
    env.close()
    return episode_times, cumulative_times, rewards

def run_benchmarks():
    print("===== RUNNING TEMPORAL SPEED BENCHMARKS (10 SEEDS, 5000 EPISODES) =====")
    
    results = {}
    
    for name in CONFIGS:
        print(f"Starting benchmark for: {name}")
        tasks = []
        for seed in SEEDS:
            tasks.append((name, seed))
            
        seed_results = []
        max_workers = min(len(tasks), os.cpu_count() or 4)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_single_seed, n, s): (n, s) for n, s in tasks}
            for future in as_completed(futures):
                n, s = futures[future]
                try:
                    ep_times, cum_times, rewards = future.result()
                    seed_results.append({
                        "seed": s,
                        "episode_times": ep_times,
                        "cumulative_times": cum_times,
                        "rewards": rewards
                    })
                except Exception as e:
                    print(f"Error running {n} seed {s}: {e}")
                    
        # Average across the 10 seeds
        all_ep_times = np.array([res["episode_times"] for res in seed_results])
        all_cum_times = np.array([res["cumulative_times"] for res in seed_results])
        all_rewards = np.array([res["rewards"] for res in seed_results])
        
        results[name] = {
            "mean_episode_times": all_ep_times.mean(axis=0).tolist(),
            "mean_cumulative_times": all_cum_times.mean(axis=0).tolist(),
            "mean_rewards": all_rewards.mean(axis=0).tolist(),
            # Save raw data for convergence threshold checks per seed
            "seeds": [
                {
                    "cumulative_times": res["cumulative_times"],
                    "rewards": res["rewards"]
                } for res in seed_results
            ]
        }
        
    # Save raw benchmark data
    os.makedirs("results", exist_ok=True)
    with open("results/speed_benchmark_data.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print("Benchmark data saved successfully.")
    
    # Generate Plots
    generate_plots(results)

def calculate_convergence_time(seed_rewards, seed_cum_times, threshold):
    rolling_avg = pd.Series(seed_rewards).rolling(window=100).mean().fillna(0).values
    cross_indices = np.where(rolling_avg >= threshold)[0]
    if len(cross_indices) == 0:
        return seed_cum_times[-1] # Return total elapsed if it never converges
    conv_ep = cross_indices[0]
    return seed_cum_times[conv_ep]

def generate_plots(results):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # --- PLOT 1: TIME PER EPISODE (1x2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    window = 100
    
    # 4x4 Grid
    q_4x4_smooth = pd.Series(results["QUEST_4x4"]["mean_episode_times"]).rolling(window=window, min_periods=1).mean() * 1000
    r_4x4_smooth = pd.Series(results["RMax_4x4"]["mean_episode_times"]).rolling(window=window, min_periods=1).mean() * 1000
    
    ax1.plot(q_4x4_smooth, color='crimson', linewidth=2, label='QUEST')
    ax1.plot(r_4x4_smooth, color='royalblue', linewidth=2, label='R-Max')
    ax1.set_title("4x4 Grid: Wall-Clock Time per Episode", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Duration (ms, Smoothed)")
    ax1.legend()
    
    # 8x8 Grid
    q_8x8_smooth = pd.Series(results["QUEST_8x8"]["mean_episode_times"]).rolling(window=window, min_periods=1).mean() * 1000
    r_8x8_smooth = pd.Series(results["RMax_8x8"]["mean_episode_times"]).rolling(window=window, min_periods=1).mean() * 1000
    
    ax2.plot(q_8x8_smooth, color='crimson', linewidth=2, label='QUEST')
    ax2.plot(r_8x8_smooth, color='royalblue', linewidth=2, label='R-Max')
    ax2.set_title("8x8 Grid: Wall-Clock Time per Episode", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Duration (ms, Smoothed)")
    ax2.legend()
    
    fig.suptitle("Wall-Clock Time per Episode over 5,000 Episodes (Mean of 10 Seeds)", fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig("results/speed_time_per_episode.png", bbox_inches='tight')
    print("Created: results/speed_time_per_episode.png")
    
    # --- PLOT 2: TIME ELAPSED AT CONVERGENCE (Bar Chart) ---
    # Compute convergence times for each seed
    conv_times = {name: [] for name in CONFIGS}
    
    for name in CONFIGS:
        threshold = 0.70 if "4x4" in name else 0.85
        for seed_data in results[name]["seeds"]:
            t_conv = calculate_convergence_time(seed_data["rewards"], seed_data["cumulative_times"], threshold)
            conv_times[name].append(t_conv)
            
    # Calculate means and standard errors
    means = {name: np.mean(conv_times[name]) for name in CONFIGS}
    stds = {name: np.std(conv_times[name]) / np.sqrt(NUM_SEEDS) for name in CONFIGS}
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    categories = ['4x4 Grid (70% Conv)', '8x8 Grid (85% Conv)']
    x = np.arange(len(categories))
    width = 0.35
    
    quest_means = [means["QUEST_4x4"], means["QUEST_8x8"]]
    quest_stds = [stds["QUEST_4x4"], stds["QUEST_8x8"]]
    
    rmax_means = [means["RMax_4x4"], means["RMax_8x8"]]
    rmax_stds = [stds["RMax_4x4"], stds["RMax_8x8"]]
    
    # Plot bars without error bars
    bar1 = ax.bar(x - width/2, quest_means, width, color='crimson', label='QUEST', edgecolor='black', alpha=0.9)
    bar2 = ax.bar(x + width/2, rmax_means, width, color='royalblue', label='R-Max', edgecolor='black', alpha=0.9)
    
    # Add values on top of bars
    for bar in bar1 + bar2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    ax.set_ylabel('Wall-Clock Time Elapsed (Seconds, Mean of 10 Seeds)', fontsize=11, fontweight='bold')
    ax.set_title('Real Wall-Clock Time Required to Reach Convergence', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(0, max(max(quest_means), max(rmax_means)) * 1.15)
    
    fig.tight_layout()
    fig.savefig("results/speed_time_to_convergence.png", bbox_inches='tight')
    print("Created: results/speed_time_to_convergence.png")

if __name__ == "__main__":
    run_benchmarks()
