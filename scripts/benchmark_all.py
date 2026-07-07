# benchmark_all.py
import time
import gymnasium as gym
from agents.quest_agent import QUESTAgent
from agents.rmax_agent import RMaxAgent

def run_quest_benchmark(env_name, map_name, max_steps, config):
    env = gym.make(env_name, is_slippery=True, map_name=map_name, max_episode_steps=max_steps)
    agent = QUESTAgent(env.observation_space, env.action_space, config)
    
    start_time = time.time()
    for episode in range(2000):
        state, info = env.reset(seed=episode)
        terminated, truncated = False, False
        for step in range(max_steps):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.learn(state, action, reward, new_state)
            state = new_state
            if terminated or truncated:
                break
        agent.on_episode_end(episode, 0)
    duration = time.time() - start_time
    env.close()
    return duration

def run_rmax_benchmark(env_name, map_name, max_steps, config):
    env = gym.make(env_name, is_slippery=True, map_name=map_name, max_episode_steps=max_steps)
    agent = RMaxAgent(env.observation_space, env.action_space, config)
    
    start_time = time.time()
    for episode in range(2000):
        state, info = env.reset(seed=episode)
        terminated, truncated = False, False
        for step in range(max_steps):
            action = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.learn(state, action, reward, new_state, terminated)
            state = new_state
            if terminated or truncated:
                break
        agent.on_episode_end(episode, 0)
    duration = time.time() - start_time
    env.close()
    return duration

def main():
    # 4x4 Configs
    quest_4x4_config = {
        "discount_rate": 0.9889047072670322,
        "exploration_constant_c": 0.08240426095596366,
        "max_iterations_multiplier": 5 # 5x the number of states
    }
    rmax_4x4_config = {
        "discount_rate": 0.9849838462915839,
        "m": 14
    }
    
    # 8x8 Configs
    quest_8x8_config = {
        "discount_rate": 0.9988452099891397,
        "exploration_constant_c": 0.01380488848357483,
        "max_iterations_multiplier": 5 # 5x the number of states
    }
    rmax_8x8_config = {
        "discount_rate": 0.9989859250982733,
        "m": 8
    }
    
    print("Benchmarking 4x4...")
    t_quest_4x4 = run_quest_benchmark("FrozenLake-v1", "4x4", 100, quest_4x4_config)
    t_rmax_4x4 = run_rmax_benchmark("FrozenLake-v1", "4x4", 100, rmax_4x4_config)
    
    print("Benchmarking 8x8...")
    t_quest_8x8 = run_quest_benchmark("FrozenLake-v1", "8x8", 200, quest_8x8_config)
    t_rmax_8x8 = run_rmax_benchmark("FrozenLake-v1", "8x8", 200, rmax_8x8_config)
    
    print("\n" + "="*40)
    print(" WALL CLOCK TIME COMPARISON (2000 Episodes)")
    print("="*40)
    print(f"| Grid Size | QUEST (5x multiplier) | R-Max | Ratio (QUEST / R-Max) |")
    print(f"|-----------|------------------------|-------|------------------------|")
    print(f"|    4x4    |         {t_quest_4x4:.2f}s         | {t_rmax_4x4:.2f}s |         {t_quest_4x4/t_rmax_4x4:.2f}x          |")
    print(f"|    8x8    |         {t_quest_8x8:.2f}s         | {t_rmax_8x8:.2f}s |         {t_quest_8x8/t_rmax_8x8:.2f}x          |")
    print("="*40)

if __name__ == "__main__":
    main()
