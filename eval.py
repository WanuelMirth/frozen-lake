import gymnasium as gym
import numpy as np
import time
import os
import json
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import random

from agents.uct_rbql_agent import UCTRBQLAgent

# --- CONFIGURATION ---
BASE_CONFIG = {
    "env_name": "FrozenLake-v1",
    "is_slippery": True,
    "map_name": "4x4",
    "agent": "RMB-Q",
    "total_episodes": 2000,   # Sufficient for 4x4 convergence
    "max_steps_per_episode": 200,
    "discount_rate": 0.99,
    "render": False,
    "seed": None,            # None = Random seed per trial
}

def run_single_trial(config, seed):
    """
    Runs a single training loop for one seed.
    Returns: List of rewards per episode.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    env = gym.make(
        config["env_name"], 
        is_slippery=config["is_slippery"], 
        map_name=config["map_name"]
    )
    
    agent = UCTRBQLAgent(env.observation_space, env.action_space, config)
    rewards = []
    
    for episode in range(config["total_episodes"]):
        # Ensure episode variability even with fixed trial seed
        current_seed = seed + episode if seed is not None else None
        state, _ = env.reset(seed=current_seed)
        
        episode_reward = 0
        terminated, truncated = False, False
        
        for step in range(config["max_steps_per_episode"]):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            agent.learn(state, action, reward, new_state)
            state = new_state
            
            if terminated or truncated:
                break
        
        agent.on_episode_end(episode, episode_reward)
        rewards.append(episode_reward)
    
    env.close()
    return rewards

def evaluate_c_value(c_val, num_seeds=10):
    """
    Runs the experiment for a specific 'c' across multiple seeds.
    Returns:
      - avg_final_score: Mean success rate of last 100 episodes across all seeds.
      - avg_convergence_speed: Mean episode number where success rate hit 70%.
    """
    print(f"\n--- Evaluating c = {c_val} ({num_seeds} Seeds) ---")
    
    config = BASE_CONFIG.copy()
    config["exploration_constant_c"] = c_val
    
    all_rewards_matrix = [] # Shape: (num_seeds, total_episodes)
    convergence_episodes = []
    
    for i in range(num_seeds):
        # Use a random seed for each trial to ensure robustness
        seed = random.randint(0, 999999)
        rewards = run_single_trial(config, seed)
        all_rewards_matrix.append(rewards)
        
        # Calculate convergence for this seed
        # (Episode where rolling avg of last 100 first crosses 0.70)
        rolling_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        convergence_idx = np.where(rolling_avg >= 0.70)[0]
        
        if len(convergence_idx) > 0:
            conv_ep = convergence_idx[0]
        else:
            conv_ep = config["total_episodes"] # Did not converge
            
        convergence_episodes.append(conv_ep)
        print(f"  Seed {i+1}/{num_seeds} | Conv: {conv_ep} | Final: {np.mean(rewards[-100:]):.2f}")

    # Aggregation
    avg_final_score = np.mean([np.mean(r[-100:]) for r in all_rewards_matrix])
    avg_convergence = np.mean(convergence_episodes)
    
    return avg_final_score, avg_convergence

def run_sensitivity_sweep():
    """
    Sweeps over different 'c' values to generate the performance curve.
    """
    # Define the range of 'c' values to test
    c_values = [0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2]
    
    results = {
        "c": [],
        "performance": [],
        "speed": []
    }
    
    print(f"Starting Sensitivity Sweep for c values: {c_values}")
    
    for c in c_values:
        perf, speed = evaluate_c_value(c, num_seeds=10) # 15 seeds per C value
        results["c"].append(c)
        results["performance"].append(perf)
        results["speed"].append(speed)
    
    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: Convergence Speed (Lower is better)
    color = 'tab:red'
    ax1.set_xlabel('Exploration Constant (c)')
    ax1.set_ylabel('Episodes to Convergence (Target: 70%)', color=color)
    ax1.plot(results["c"], results["speed"], color=color, marker='o', linestyle='dashed', label='Convergence Speed')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.invert_yaxis() # Invert because fewer episodes = faster speed

    # Plot 2: Final Performance (Higher is better)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Final Success Rate (Avg)', color=color)
    ax2.plot(results["c"], results["performance"], color=color, marker='s', label='Success Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('RMB-Q Sensitivity Analysis: Impact of C-Value')
    fig.tight_layout()
    
    filename = f"results/c_sensitivity_curve_{datetime.now().strftime('%H%M')}.png"
    plt.savefig(filename)
    print(f"\nSensitivity curve saved to {filename}")

if __name__ == "__main__":
    # OPTION 1: Run just your specific request (0.05 with 15 seeds)
    # score, speed = evaluate_c_value(0.05, num_seeds=15)
    # print(f"Result: Score={score:.2f}, ConvSpeed={speed:.1f}")
    
    # OPTION 2: Run the full curve generation (Includes 0.05)
    # This will take longer but gives you the graph you asked for.
    run_sensitivity_sweep()