# benchmark_wall_clock.py
import time
import gymnasium as gym
from agents.quest_agent import QUESTAgent
from agents.rmax_agent import RMaxAgent
import numpy as np

def benchmark():
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", max_episode_steps=200)
    
    # 1. QUEST Configuration (Trial 99)
    quest_config = {
        "discount_rate": 0.9988,
        "exploration_constant_c": 0.0138,
        "max_iterations_multiplier": 5
    }
    
    # 2. R-Max Configuration
    rmax_config = {
        "discount_rate": 0.9990,
        "m": 8
    }
    
    print("=== Running QUEST Benchmark (2000 episodes) ===")
    quest_agent = QUESTAgent(env.observation_space, env.action_space, quest_config)
    
    start_time = time.time()
    for episode in range(2000):
        state, info = env.reset(seed=episode)
        terminated, truncated = False, False
        for step in range(200):
            action = quest_agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            quest_agent.learn(state, action, reward, new_state)
            state = new_state
            if terminated or truncated:
                break
        quest_agent.on_episode_end(episode, 0)
    quest_duration = time.time() - start_time
    print(f"QUEST completed in {quest_duration:.2f} seconds.")
    
    print("\n=== Running R-Max Benchmark (2000 episodes) ===")
    rmax_agent = RMaxAgent(env.observation_space, env.action_space, rmax_config)
    
    start_time = time.time()
    for episode in range(2000):
        state, info = env.reset(seed=episode)
        terminated, truncated = False, False
        for step in range(200):
            action = rmax_agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            rmax_agent.learn(state, action, reward, new_state, terminated)
            state = new_state
            if terminated or truncated:
                break
        rmax_agent.on_episode_end(episode, 0)
    rmax_duration = time.time() - start_time
    print(f"R-Max completed in {rmax_duration:.2f} seconds.")
    
    print(f"\nSpeed comparison: R-Max is {quest_duration / rmax_duration:.2f}x faster than QUEST.")

if __name__ == "__main__":
    benchmark()
