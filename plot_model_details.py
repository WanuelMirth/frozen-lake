# plot_model_details.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json

# FrozenLake 4x4 layout mapping
GRID_SIZE = 4
HOLES = {5, 7, 11, 12}
GOAL = 15
START = 0
JSON_PATH = "results/q_table_snapshots.json"
SEEDS = [42, 43, 44]

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
        
    return all_seeds_data

def plot_model_coverage(snapshots, checkpoints, first_goal_episode, seed, save_path):
    # Setup a 3x4 grid (12 subplots total)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=300)
    axes = axes.flatten()
    
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
    
    ax_env.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_env.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax_env.set_xticks(np.arange(GRID_SIZE))
    ax_env.set_yticks(np.arange(GRID_SIZE))
    ax_env.set_xticklabels([])
    ax_env.set_yticklabels([])
    ax_env.tick_params(axis='both', which='both', length=0)
    ax_env.set_xticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
    ax_env.set_yticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
    ax_env.grid(False, which='major')
    ax_env.grid(True, which='minor', color='white', linestyle='-', linewidth=2.5, zorder=5)
    
    # 2. Plot check-points in the remaining slots
    for i, ep in enumerate(checkpoints, start=1):
        if i >= len(axes):
            break
        ax = axes[i]
        _, _, n_sa, model = snapshots[ep]
        
        # Calculate transition arrivals to identify undiscovered states
        k_counts = np.zeros(16)
        for s_src in model:
            for a in model[s_src]:
                for s_next in model[s_src][a]:
                    k_counts[s_next] += model[s_src][a][s_next]
        
        # Calculate coverage grid (number of explored actions out of 4)
        cov_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for s in range(16):
            r, c = divmod(s, GRID_SIZE)
            if s != START and k_counts[s] == 0:
                cov_grid[r, c] = np.nan # Undiscovered
            elif s in HOLES:
                cov_grid[r, c] = -1.0 # Set hole indicator (under vmin)
            elif s == GOAL:
                cov_grid[r, c] = 5.0 # Goal has 0 outgoing actions (over vmax)
            else:
                cov_grid[r, c] = sum(1 for a in range(4) if n_sa[s][a] > 0)
                
        # Draw heatmap (sliced Purples colormap with bad values colored light grey)
        orig_cmap = plt.cm.Purples
        colors = orig_cmap(np.linspace(0.2, 1.0, 256))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("sliced_purples", colors)
        cmap.set_under(color='black')
        cmap.set_over(color=goal_layout_color)
        cmap.set_bad(color='#f5f5f5')
        im = ax.imshow(cov_grid, cmap=cmap, vmin=0, vmax=4)
        
        title_suffix = ""
        if ep == first_goal_episode:
            title_suffix = " (First Goal)"
        ax.set_title(f"Episode {ep}{title_suffix}", fontsize=13, fontweight='bold', pad=10)
        
        # Annotate
        for s in range(16):
            r, c = divmod(s, GRID_SIZE)
            val = cov_grid[r, c]
            
            if np.isnan(val):
                ax.text(c, r, "?", ha="center", va="center", fontsize=16, color="black", fontweight='bold')
            elif s in HOLES:
                ax.text(c, r, "H", ha="center", va="center", fontsize=14, color="white", fontweight='bold')
            elif s == GOAL:
                ax.text(c, r, "G", ha="center", va="center", fontsize=14, color="white", fontweight='bold')
            else:
                cov_val = int(val)
                text_str = f"{cov_val}/4"
                text_color = "white" if cov_val >= 3 else "black"
                ax.text(c, r, text_str, ha="center", va="center", fontsize=11, color=text_color, fontweight='semibold')
                
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(GRID_SIZE - 0.5, -0.5)
        ax.set_xticks(np.arange(GRID_SIZE))
        ax.set_yticks(np.arange(GRID_SIZE))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        
        ax.set_xticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(GRID_SIZE + 1) - 0.5, minor=True)
        ax.grid(False, which='major')
        ax.grid(True, which='minor', color='white', linestyle='-', linewidth=2.5, zorder=5)
    # (suptitle removed)
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.9, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Explored Actions (0 to 4)")
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Model coverage plot saved successfully to: {save_path}")
    plt.close()

def plot_slippage_learning(snapshots, checkpoints, first_goal_episode, seed, save_path):
    # We track State 10, Action 1 (Down)
    target_state = 10
    target_action = 1
    
    # Outcomes:
    # 14 is Intended (Down)
    # 9 is Slip Left
    # 11 is Slip Right
    outcomes = [14, 9, 11]
    outcome_labels = ["Down (Intended)", "Left (Slip Left)", "Right (Slip Right)"]
    colors = ["#d32f2f", "#1976d2", "#f57c00"]
    
    # Gather data across checkpoints
    groups = ["Ground Truth"]
    # Grouped data: list of lists [down_prob, left_prob, right_prob]
    data_probs = [[1/3, 1/3, 1/3]]
    
    for ep in checkpoints:
        groups.append(f"Ep {ep}" + ("\n(First Goal)" if ep == first_goal_episode else ""))
        _, _, _, model = snapshots[ep]
        
        counts = model.get(target_state, {}).get(target_action, {})
        total = sum(counts.values())
        
        if total == 0:
            data_probs.append([0.0, 0.0, 0.0])
        else:
            probs = [
                counts.get(14, 0) / total,
                counts.get(9, 0) / total,
                counts.get(11, 0) / total
            ]
            data_probs.append(probs)
            
    # Draw grouped bar chart
    x = np.arange(len(groups))
    width = 0.23
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot bars
    data_probs = np.array(data_probs)
    ax.bar(x - width, data_probs[:, 0], width, label=outcome_labels[0], color=colors[0], edgecolor='black', alpha=0.9)
    ax.bar(x, data_probs[:, 1], width, label=outcome_labels[1], color=colors[1], edgecolor='black', alpha=0.9)
    ax.bar(x + width, data_probs[:, 2], width, label=outcome_labels[2], color=colors[2], edgecolor='black', alpha=0.9)
    
    ax.set_title(f"Slippage Model Learning over Checkpoints (Seed {seed})\nTarget: State {target_state} | Action: Down", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Checkpoints / Matures", fontsize=11, fontweight='semibold')
    ax.set_ylabel("Transition Probability $P(s' \\mid s, a)$", fontsize=11, fontweight='semibold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10, fontweight='semibold')
    ax.set_ylim(0.0, 1.05)
    
    # Add values on top of bars
    for i in range(len(groups)):
        for j in range(3):
            val = data_probs[i, j]
            if val > 0.0:
                offset = -width if j == 0 else (0 if j == 1 else width)
                ax.text(i + offset, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
                
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Slippage learning plot saved successfully to: {save_path}")
    plt.close()

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Cache file '{JSON_PATH}' not found. Please run plot_state_values.py first to generate it.")
        return
        
    print("Loading cached training data for all seeds...")
    all_seeds_data = load_all_seeds_from_json(JSON_PATH)
    
    os.makedirs("results", exist_ok=True)
    
    for seed in SEEDS:
        if seed in all_seeds_data:
            snapshots, checkpoints, first_goal_episode = all_seeds_data[seed]
            
            # Select 3x4 display subset for coverage plots
            map_cov_checkpoints = {1, 5, 10, 20, 30, 40, 50, 100, 250, 1000}
            if first_goal_episode is not None:
                map_cov_checkpoints.add(first_goal_episode)
            map_cov_checkpoints = sorted(list(map_cov_checkpoints))
            
            # Select 2x3 display subset for slippage plots
            slip_checkpoints = {1, 30, 250, 1000}
            if first_goal_episode is not None:
                slip_checkpoints.add(first_goal_episode)
            slip_checkpoints = sorted(list(slip_checkpoints))
            
            # 1. Plot Model Coverage Grid
            cov_path = f"results/model_coverage_seed{seed}.png"
            plot_model_coverage(snapshots, map_cov_checkpoints, first_goal_episode, seed, cov_path)
            
            # 2. Plot Slippage Learning Bar Chart
            slip_path = f"results/slippage_learning_seed{seed}.png"
            plot_slippage_learning(snapshots, slip_checkpoints, first_goal_episode, seed, slip_path)
        else:
            print(f"Warning: Seed {seed} not found in cache data.")

if __name__ == "__main__":
    main()
