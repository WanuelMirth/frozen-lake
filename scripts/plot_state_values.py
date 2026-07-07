# plot_state_values.py
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random
import json

from agents.quest_agent import QUESTAgent

# FrozenLake 4x4 layout mapping
GRID_SIZE = 4
HOLES = {5, 7, 11, 12}
GOAL = 15
START = 0

JSON_PATH = "results/q_table_snapshots.json"
SEEDS = [42, 43, 44]

def run_training_and_capture_snapshots(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    
    env = gym.make(
        config["env_name"], 
        is_slippery=config["is_slippery"], 
        map_name=config["map_name"]
    )
    
    # Make a copy of the config and inject seed for agent
    agent_config = config.copy()
    agent_config["seed"] = seed
    agent = QUESTAgent(env.observation_space, env.action_space, agent_config)
    
    snapshots = {}
    first_goal_episode = None
    
    total_episodes = 1000
    
    for episode in range(1, total_episodes + 1):
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
        
        # Track first goal reach
        if episode_reward > 0 and first_goal_episode is None:
            first_goal_episode = episode
            print(f"Seed {seed}: Goal reached for the first time in Episode {first_goal_episode}!")
            
        # Capture snapshot for every episode
        q_table_snap = {s: np.copy(agent._get_q_values(s)) for s in range(16)}
        n_s_snap = {s: agent.n_s.get(s, 0) for s in range(16)}
        n_sa_snap = {s: np.copy(agent.n_sa.get(s, np.zeros(agent.n_actions))) for s in range(16)}
        model_snap = {}
        for s in agent.model:
            model_snap[s] = {}
            for a in range(agent.n_actions):
                model_snap[s][a] = {ns: count for ns, count in agent.model[s][a].items()}
        snapshots[episode] = (q_table_snap, n_s_snap, n_sa_snap, model_snap)
            
    env.close()
    
    # Store checkpoints as 1 to 1000
    final_checkpoints = list(range(1, total_episodes + 1))
    return snapshots, final_checkpoints, first_goal_episode

def save_all_seeds_to_json(all_seeds_data, path):
    serializable = {}
    for seed, seed_data in all_seeds_data.items():
        snapshots, checkpoints, first_goal_episode = seed_data
        serializable_snapshots = {}
        for ep, (q_table, n_s, n_sa, model) in snapshots.items():
            serializable_snapshots[str(ep)] = {
                "q_table": {str(s): q_table[s].tolist() for s in range(16)},
                "n_s": {str(s): int(n_s[s]) for s in range(16)},
                "n_sa": {str(s): n_sa[s].tolist() for s in range(16)},
                "model": {str(s): {str(a): {str(ns): count for ns, count in model.get(s, {}).get(a, {}).items()}
                                   for a in range(4)}
                          for s in model}
            }
        serializable[str(seed)] = {
            "first_goal_episode": first_goal_episode,
            "checkpoints": checkpoints,
            "snapshots": serializable_snapshots
        }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=4)
    print(f"All seeds saved to cache at '{path}'")

def load_all_seeds_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
        
    all_seeds_data = {}
    for seed_str, seed_data in data.items():
        seed = int(seed_str)
        first_goal_episode = seed_data["first_goal_episode"]
        checkpoints = seed_data["checkpoints"]
        snapshots = {}
        for ep_str, snap in seed_data["snapshots"].items():
            ep = int(ep_str)
            q_table = {int(s): np.array(vals) for s, vals in snap["q_table"].items()}
            n_s = {int(s): val for s, val in snap["n_s"].items()}
            
            n_sa_raw = snap.get("n_sa", {})
            n_sa = {int(s): np.array(vals) for s, vals in n_sa_raw.items()}
            
            model_raw = snap.get("model", {})
            model = {int(s): {int(a): {int(ns): count for ns, count in model_raw[s][a].items()}
                              for a in model_raw[s]}
                     for s in model_raw}
            
            snapshots[ep] = (q_table, n_s, n_sa, model)
        all_seeds_data[seed] = (snapshots, checkpoints, first_goal_episode)
        
    print(f"All seeds loaded from cache at '{path}'")
    return all_seeds_data

def plot_snapshots(snapshots, checkpoints, first_goal_episode, seed, save_path):
    # Setup a 3x4 grid (12 subplots total)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=300)
    axes = axes.flatten()
    
    arrow_symbols = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    
    # Get exact Goal color from the Reds colormap at value 1.0
    reds_cmap = plt.cm.Reds
    goal_layout_color = reds_cmap(1.0)
    
    # 1. Plot the Environment Layout in the first slot
    ax_env = axes[0]
    
    # Create layout color map: 0=Frozen, 1=Start, 2=Goal, 3=Hole
    layout_colors = ['#f5f5f5', 'green', goal_layout_color, 'black']
    layout_cmap = matplotlib.colors.ListedColormap(layout_colors)
    
    env_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for s in range(16):
        r, c = divmod(s, GRID_SIZE)
        if s == START:
            env_grid[r, c] = 1
        elif s == GOAL:
            env_grid[r, c] = 2
        elif s in HOLES:
            env_grid[r, c] = 3
        else:
            env_grid[r, c] = 0
            
    ax_env.imshow(env_grid, cmap=layout_cmap, vmin=0, vmax=3)
    ax_env.set_title("Environment Map Layout", fontsize=13, fontweight='bold', pad=10)
    
    for s in range(16):
        r, c = divmod(s, GRID_SIZE)
        if s == START:
            label, color = "S", "white"
        elif s == GOAL:
            label, color = "G", "white"
        elif s in HOLES:
            label, color = "H", "white"
        else:
            label, color = "F", "black"
        ax_env.text(c, r, label, ha="center", va="center", fontsize=16, color=color, fontweight='bold')
    
    # Format environment layout axis
    ax_env.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_env.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax_env.set_xticks(np.arange(GRID_SIZE))
    ax_env.set_yticks(np.arange(GRID_SIZE))
    ax_env.set_xticklabels([])
    ax_env.set_yticklabels([])
    
    # Remove tick marks
    ax_env.tick_params(axis='both', which='both', length=0)
    
    # Correct grid lines at the boundaries (minor grid lines)
    ax_env.set_xticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
    ax_env.set_yticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
    ax_env.grid(False, which='major')
    ax_env.grid(True, which='minor', color='white', linestyle='-', linewidth=2.5, zorder=5)
    
    # 2. Plot check-points in the remaining slots
    for i, ep in enumerate(checkpoints, start=1):
        if i >= len(axes):
            break
        ax = axes[i]
        q_table, n_s = snapshots[ep][:2]
        
        # Calculate State Value V(s) = max_a Q(s,a)
        v_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for s in range(16):
            r, c = divmod(s, GRID_SIZE)
            if s in HOLES:
                v_grid[r, c] = -1.0 # Hole indicator (under vmin)
            elif s == GOAL:
                v_grid[r, c] = 5.0 # Goal indicator (over vmax)
            else:
                v_grid[r, c] = np.max(q_table[s])
                
        # Draw heatmap with standard Reds colormap
        cmap = plt.cm.Reds.copy()
        cmap.set_under(color='black')
        cmap.set_over(color=goal_layout_color)
        im = ax.imshow(v_grid, cmap=cmap, vmin=0.0, vmax=1.0)
        
        title_suffix = ""
        if ep == first_goal_episode:
            title_suffix = " (First Goal)"
        ax.set_title(f"Episode {ep}{title_suffix}", fontsize=13, fontweight='bold', pad=10)
        
        # Annotate cell values and arrows
        for s in range(16):
            r, c = divmod(s, GRID_SIZE)
            
            if s in HOLES:
                ax.text(c, r, "H", ha="center", va="center", fontsize=14, color="white", fontweight='bold')
            elif s == GOAL:
                ax.text(c, r, "G", ha="center", va="center", fontsize=14, color="white", fontweight='bold')
            else:
                val = v_grid[r, c]
                q_vals = q_table[s]
                
                # Check if state has been visited and has non-zero Q-value
                if n_s[s] > 0 and np.any(q_vals > 0):
                    best_action = np.argmax(q_vals)
                    arrow = arrow_symbols[best_action]
                    text_str = f"{val:.2f}\n{arrow}"
                    # Contrast color depending on value: high (red) is white text, low (grey) is black text
                    text_color = "white" if val > 0.5 else "black"
                else:
                    text_str = f"{val:.2f}\n•"
                    text_color = "black"
                    
                ax.text(c, r, text_str, ha="center", va="center", fontsize=10, color=text_color, fontweight='semibold')
                
        # Format axes
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Remove tick marks
        ax.tick_params(axis='both', which='both', length=0)
        
        # Minor grid lines for correct cell boundaries
        ax.set_xticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
        ax.grid(False, which='major')
        ax.grid(True, which='minor', color='white', linestyle='-', linewidth=2.5, zorder=5)
        
    # Main styling and colorbar (suptitle removed)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="State Value V(s)")
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Progression plot saved successfully to: {save_path}")

def main():
    # CHOOSE SEED HERE: 42, 43, or 44
    SELECTED_SEED = 42
    
    config = {
        "env_name": "FrozenLake-v1",
        "is_slippery": True,
        "map_name": "4x4",
        "max_steps_per_episode": 200,
        "discount_rate": 0.9843205299768125,       # Trial 92 parameters
        "exploration_constant_c": 0.09900852192118169,
    }
    
    required_base_len = 1000
    use_cache = False
    
    if os.path.exists(JSON_PATH):
        try:
            all_seeds_data = load_all_seeds_from_json(JSON_PATH)
            if all(s in all_seeds_data for s in SEEDS):
                # Verify that cache contains 1000 checkpoints for all seeds
                sample_seed = SEEDS[0]
                if len(all_seeds_data[sample_seed][0]) >= required_base_len:
                    sample_snap = all_seeds_data[sample_seed][0][1] # Episode 1
                    if len(sample_snap) >= 4 and len(sample_snap[3]) > 0:
                        use_cache = True
                    else:
                        print("Cache has older format without transition model. Re-running training...")
                else:
                    print("Cache has incomplete checkpoints. Re-running training...")
            else:
                print("Cache is missing some seeds. Re-running training...")
        except Exception as e:
            print(f"Error loading cache: {e}. Re-running training...")
            
    if not use_cache:
        print("Gathering training data for all seeds...")
        all_seeds_data = {}
        for seed in SEEDS:
            snapshots, checkpoints, first_goal_episode = run_training_and_capture_snapshots(config, seed)
            all_seeds_data[seed] = (snapshots, checkpoints, first_goal_episode)
        save_all_seeds_to_json(all_seeds_data, JSON_PATH)
        
    # Plot comparison heatmaps for all seeds (using 3x4 layout with a subset of checkpoints)
    for seed in SEEDS:
        if seed in all_seeds_data:
            snapshots, checkpoints, first_goal_episode = all_seeds_data[seed]
            print(f"\nGenerating value progression subplots for seed {seed}...")
            
            # Select 3x4 display subset for value progression plot
            val_checkpoints = {1, 5, 10, 20, 30, 40, 50, 100, 250, 1000}
            if first_goal_episode is not None:
                val_checkpoints.add(first_goal_episode)
            val_checkpoints = sorted(list(val_checkpoints))
            
            print(f"Value Checkpoints: {val_checkpoints}")
            print(f"First Goal Episode: {first_goal_episode}")
            save_path = f"results/state_value_progression_seed{seed}.png"
            plot_snapshots(snapshots, val_checkpoints, first_goal_episode, seed, save_path)
        else:
            print(f"Error: seed {seed} not found in gathered data.")

if __name__ == "__main__":
    main()
