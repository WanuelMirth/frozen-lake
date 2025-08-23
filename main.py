# main.py
import gymnasium as gym
import numpy as np
import time
import os
import json
import csv
from datetime import datetime
import random

# Importiere ALLE Agenten, die wir jemals verwenden wollen
from agents.q_learning_agent import QLearningAgent
from agents.dyna_t_agent import DynaTAgent
from agents.stochastic_rbql_agent import StochasticRBQLAgent
from agents.uct_rbql_agent import UCTRBQLAgent


from utils import plot_results

# --- Konfigurations-Bibliothek für EINZELNE Läufe ---
# Dient als Referenz und für schnelle Tests. Die Sweeps definieren ihre eigenen Konfigurationen.
CONFIGS = {
    "Dyna_T_4x4_Slippery_CHAMPION": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "Dyna-T", "total_episodes": 5000, "max_steps_per_episode": 200,
        "learning_rate": 0.05, "planning_steps": 50, "exploration_constant_c": 0.2,
        "discount_rate": 0.99, "render": False,
    },
    "StochasticRBQL2_4x4_Slippery_CHAMPION": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "StochasticRBQL",
        "total_episodes": 5000, "max_steps_per_episode": 200,
        "discount_rate": 0.99, "max_epsilon": 1.0, "min_epsilon": 0.05,
        "epsilon_decay_rate": 0.0005,
        "render": False,
    },
    "UCT_RBQL_4x4_Slippery": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "UCT-RBQL",  # Wähle den neuen Agenten
        "total_episodes": 5000, "max_steps_per_episode": 200,
        "discount_rate": 0.99,
        "exploration_constant_c": 0.1,
        "render": False,
    },
}


def setup_experiment(run_name, config):
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    csv_file = open(f"{results_dir}/metrics.csv", 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'steps', 'reward'])
    print(f"Starte Experiment: {run_name}")
    print(f"Ergebnisse werden in '{results_dir}' gespeichert.")
    return results_dir, csv_writer, csv_file

def train(run_name, config):
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
    
    # --- ERWEITERTE AGENTEN-AUSWAHL ---
    agent_name = config["agent"]
    if agent_name == "Dyna-T":
        agent = DynaTAgent(env.observation_space, env.action_space, config)
    elif agent_name == "StochasticRBQL":
        agent = StochasticRBQLAgent(env.observation_space, env.action_space, config)
    elif agent_name == "UCT-RBQL":
        agent = UCTRBQLAgent(env.observation_space, env.action_space, config)
    elif agent_name == "Q-Learning":
        agent = QLearningAgent(env.observation_space, env.action_space, config)
    else:
        raise ValueError(f"Unbekannter Agent in Konfiguration: {agent_name}")
        
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
            print(f"  {run_name} - Episode {episode + 1}: Avg Reward (last 100) = {avg_reward:.3f}")

            
    env.close()
    csv_file.close()
    
    end_time = time.time()
    duration_seconds = end_time - start_time
    
    print(f"--- Training für {run_name} abgeschlossen in {duration_seconds:.2f} Sekunden ---")
    plot_results(results_dir, config)
    
    return results_dir, duration_seconds

if __name__ == "__main__":
    # Dieser Block ist jetzt nur für schnelle, einzelne Testläufe
    config_name = "UCT_RBQL_4x4_Slippery" # Beispiel
    config_to_run = CONFIGS[config_name]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{config_name}_{timestamp}"
    train(run_name, config_to_run)