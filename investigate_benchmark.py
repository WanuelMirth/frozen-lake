# investigate_benchmark.py
import gymnasium as gym
from agents.quest_agent import QUESTAgent
from agents.rmax_agent import RMaxAgent
import numpy as np

def investigate():
    env = gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8", max_episode_steps=200)
    
    # QUEST config (best candidate)
    quest_config = {
        "discount_rate": 0.9988,
        "exploration_constant_c": 0.0138,
        "max_iterations_multiplier": 5
    }
    
    # R-Max config
    rmax_config = {
        "discount_rate": 0.9990,
        "m": 8
    }
    
    # --- QUEST ---
    quest_agent = QUESTAgent(env.observation_space, env.action_space, quest_config)
    quest_rewards = []
    quest_vi_loops = 0
    
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
        
        # Count iterations in _learn_backwards manually
        all_known_states = list(quest_agent.model.keys())
        if all_known_states:
            max_iterations = len(all_known_states) * 5
            # We estimate the number of state-action updates
            quest_vi_loops += max_iterations * len(all_known_states) * 4
            
        quest_agent.on_episode_end(episode, episode_reward=reward)
        quest_rewards.append(reward)
        
    # --- R-Max ---
    rmax_agent = RMaxAgent(env.observation_space, env.action_space, rmax_config)
    rmax_rewards = []
    rmax_vi_loops = 0
    
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
                
        # Count iterations
        if rmax_agent.visited_states:
            max_iterations = len(rmax_agent.visited_states) * 10
            rmax_vi_loops += max_iterations * len(rmax_agent.visited_states) * 4
            
        rmax_agent.on_episode_end(episode, episode_reward=reward)
        rmax_rewards.append(reward)
        
    print("=== Investigation Results ===")
    print(f"QUEST:")
    print(f"  Unique states visited: {len(quest_agent.model.keys())} / 64")
    print(f"  Success rate (last 100 ep): {np.mean(quest_rewards[-100:]) * 100:.1f}%")
    print(f"  Estimated State-Action Updates: {quest_vi_loops:,}")
    
    print(f"\nR-Max:")
    print(f"  Unique states visited: {len(rmax_agent.visited_states)} / 64")
    print(f"  Success rate (last 100 ep): {np.mean(rmax_rewards[-100:]) * 100:.1f}%")
    print(f"  Estimated State-Action Updates: {rmax_vi_loops:,}")

if __name__ == "__main__":
    investigate()
