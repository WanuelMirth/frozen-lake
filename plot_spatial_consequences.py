# plot_spatial_consequences.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json

GRID_SIZE = 4
HOLES = {5, 7, 11, 12}
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

def get_ground_truth():
    # Ground Truth success (safe state) probabilities for State 6
    # Left (0): 2/3
    # Down (1): 1/3
    # Right (2): 2/3
    # Up (3): 1/3
    return {0: 2/3, 1: 1/3, 2: 2/3, 3: 1/3}

def draw_cross_subplot(ax, success_probs, unexplored, title):
    # Create 3x3 grid filled with NaN (corners transparent)
    grid = np.full((3, 3), np.nan)
    
    # Fill in the cardinal values
    grid[1, 0] = success_probs[0] # Left
    grid[2, 1] = success_probs[1] # Down
    grid[1, 2] = success_probs[2] # Right
    grid[0, 1] = success_probs[3] # Up
    
    # Colormap RdYlGn (Red = 0.0, Yellow = 0.5, Green = 1.0)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color='white')
    
    ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    # Action labels
    action_dirs = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    # Cell positions mapping (col, row)
    cell_coords = {0: (0, 1), 1: (1, 2), 2: (2, 1), 3: (1, 0)}
    
    # 1. Draw Center Cell (S6)
    rect_center = patches.Rectangle((0.5, 0.5), 1, 1, facecolor='#eeeeee', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_center)
    ax.text(1, 1, "State 6", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # 2. Draw Cardinal Cells
    for a in range(4):
        c, r = cell_coords[a]
        
        if unexplored[a]:
            # Draw unexplored cell with grey stripes
            rect_unexplored = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='#e0e0e0', hatch='//', edgecolor='black', linewidth=1.5)
            ax.add_patch(rect_unexplored)
            ax.text(c, r, f"{action_dirs[a]}\nN/A", ha='center', va='center', fontsize=9, color='gray', fontweight='bold')
        else:
            # Draw border
            rect_border = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect_border)
            
            prob = success_probs[a]
            # Choose text color (white for high contrast on deep red/green, black for light middle values)
            text_color = "white" if (prob < 0.25 or prob > 0.75) else "black"
            
            ax.text(c, r, f"{action_dirs[a]}\n{prob:.2f}", ha='center', va='center', fontsize=10, color=text_color, fontweight='bold')
            
    # Format subplot axes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(False)

def plot_spatial_consequences_for_seed(snapshots, checkpoints, first_goal_episode, seed, save_path):
    target_state = 6
    ground_truth = get_ground_truth()
    
    # 2x3 Grid layout:
    # Subplot 0: Ground Truth
    # Subplots 1-5: Checkpoints
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), dpi=300)
    axes = axes.flatten()
    
    # 1. Draw Ground Truth
    gt_probs = [ground_truth[a] for a in range(4)]
    gt_unexplored = [False] * 4
    draw_cross_subplot(axes[0], gt_probs, gt_unexplored, "Ground Truth (True Safety)")
    
    # 2. Draw Checkpoints
    for i, ep in enumerate(checkpoints, start=1):
        if i >= len(axes):
            break
        ax = axes[i]
        _, _, _, model = snapshots[ep]
        
        ch_probs = []
        ch_unexplored = []
        
        for a in range(4):
            counts = model.get(target_state, {}).get(a, {})
            total = sum(counts.values())
            
            if total == 0:
                ch_probs.append(0.0)
                ch_unexplored.append(True)
            else:
                hole_count = sum(counts.get(h, 0) for h in HOLES)
                safe_count = total - hole_count
                ch_probs.append(safe_count / total)
                ch_unexplored.append(False)
                
        title_suffix = ""
        if ep == first_goal_episode:
            title_suffix = " (First Goal)"
        draw_cross_subplot(ax, ch_probs, ch_unexplored, f"Episode {ep}{title_suffix}")
        
    plt.suptitle(f"QUEST Model learning: Safety Consequences of Actions at State 6 (Seed {seed})\n(Directions show the action taken; colors represent success probability of landing in a SAFE state)", fontsize=15, y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Spatial action consequences plot saved successfully to: {save_path}")
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
            
            # Select 2x3 display subset for spatial consequences plot
            spatial_checkpoints = {1, 30, 250, 1000}
            if first_goal_episode is not None:
                spatial_checkpoints.add(first_goal_episode)
            spatial_checkpoints = sorted(list(spatial_checkpoints))
            
            save_path = f"results/spatial_consequences_seed{seed}.png"
            plot_spatial_consequences_for_seed(snapshots, spatial_checkpoints, first_goal_episode, seed, save_path)
        else:
            print(f"Warning: Seed {seed} not found in cache data.")

if __name__ == "__main__":
    main()
