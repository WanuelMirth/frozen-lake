# main.py
import gymnasium as gym
import numpy as np
import time
import os
import json
import csv
from datetime import datetime
import random
from agents.quest_agent import QUESTAgent

def setup_experiment(run_name, config):
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    csv_file = open(f"{results_dir}/metrics.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'steps', 'reward'])
    print(f"Starting Experiment: {run_name}")
    print(f"Results will be saved in '{results_dir}'.")
    return results_dir, csv_writer, csv_file

def train(run_name, config, trial=None):
    start_time = time.time()
    
    seed = config.get('seed')
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results_dir, csv_writer, csv_file = setup_experiment(run_name, config)
    
    env = gym.make(
        config["env_name"], 
        is_slippery=config["is_slippery"], 
        map_name=config["map_name"],
        render_mode="human" if config.get("render", False) else None
    )
    
    agent = QUESTAgent(env.observation_space, env.action_space, config)
        
    rewards_per_episode = []
    
    for episode in range(config["total_episodes"]):
        current_seed = seed + episode if seed is not None else None
        state, info = env.reset(seed=current_seed)

        terminated, truncated = False, False
        episode_reward = 0
        
        for step in range(config["max_steps_per_episode"]):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            agent.learn(state, action, reward, new_state)
            state = new_state
            if config.get("render", False): time.sleep(config.get("render_sleep", 0.01))
            if terminated or truncated: break
        
        agent.on_episode_end(episode, episode_reward)
        rewards_per_episode.append(episode_reward)
        csv_writer.writerow([episode, step + 1, episode_reward])
        
        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"  {run_name} - Episode {episode + 1}: Avg Reward (last 100) = {avg_reward:.4f}")
            

    env.close()
    csv_file.close()
    
    end_time = time.time()
    duration_seconds = end_time - start_time
    
    print(f"--- Training for {run_name} completed in {duration_seconds:.2f} seconds ---")
    
    return results_dir, duration_seconds

CONFIGS = {
    "QUEST_Pareto_Trial74": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "agent": "QUEST",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9926714709783417,
        "exploration_constant_c": 0.059991060225751965,
        "render": False,
    },
    "QUEST_Pareto_Trial92": {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "agent": "QUEST",
        "total_episodes": 5000,
        "max_steps_per_episode": 200,
        "discount_rate": 0.9843205299768125,
        "exploration_constant_c": 0.09900852192118169,
        "render": False,
    }
}

if __name__ == "__main__":
    config_name = "QUEST_Pareto_Trial74"
    config_to_run = CONFIGS[config_name]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    train(run_name, config_to_run)