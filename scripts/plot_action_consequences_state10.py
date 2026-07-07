# plot_action_consequences_state10.py
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
    # Ground Truth probabilities for State 10
    # Left (0): 3/3 Safe
    # Down (1): 2/3 Safe, 1/3 Hole
    # Right (2): 2/3 Safe, 1/3 Hole
    # Up (3): 2/3 Safe, 1/3 Hole
    return {
        0: {"Safe": 1.0, "Hole": 0.0},
        1: {"Safe": 2/3, "Hole": 1/3},
        2: {"Safe": 2/3, "Hole": 1/3},
        3: {"Safe": 2/3, "Hole": 1/3}
    }

def draw_stacked_vertical_bar(ax, x_center, y_baseline, width, height, safe_prob, hole_prob, unexplored):
    x_left = x_center - width / 2
    
    if unexplored:
        # Draw hashed grey bar vertically
        rect = patches.Rectangle((x_left, y_baseline), width, height, facecolor="lightgray", hatch="//", edgecolor="gray", alpha=0.6)
        ax.add_patch(rect)
        ax.text(x_center, y_baseline + height / 2, "N/A", ha='center', va='center', color='gray', fontweight='bold', fontsize=8)
    else:
        safe_h = safe_prob * height
        hole_h = hole_prob * height
        
        # Safe part (green)
        if safe_h > 0:
            rect_safe = patches.Rectangle((x_left, y_baseline), width, safe_h, facecolor="forestgreen", edgecolor="black", alpha=0.9)
            ax.add_patch(rect_safe)
            if safe_prob > 0.05:
                ax.text(x_center, y_baseline + safe_h / 2, f"{safe_prob:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                
        # Hole part (black)
        if hole_h > 0:
            rect_hole = patches.Rectangle((x_left, y_baseline + safe_h), width, hole_h, facecolor="black", edgecolor="black", alpha=0.9)
            ax.add_patch(rect_hole)
            if hole_prob > 0.05:
                ax.text(x_center, y_baseline + safe_h + hole_h / 2, f"{hole_prob:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

def draw_stacked_horizontal_bar(ax, x_baseline, y_center, width, height, safe_prob, hole_prob, unexplored):
    y_bottom = y_center - height / 2
    
    if unexplored:
        # Draw hashed grey bar horizontally
        rect = patches.Rectangle((x_baseline, y_bottom), width, height, facecolor="lightgray", hatch="//", edgecolor="gray", alpha=0.6)
        ax.add_patch(rect)
        ax.text(x_baseline + width / 2, y_center, "N/A", ha='center', va='center', color='gray', fontweight='bold', fontsize=8)
    else:
        safe_w = safe_prob * width
        hole_w = hole_prob * width
        
        # Safe part (green)
        if safe_w > 0:
            rect_safe = patches.Rectangle((x_baseline, y_bottom), safe_w, height, facecolor="forestgreen", edgecolor="black", alpha=0.9)
            ax.add_patch(rect_safe)
            if safe_prob > 0.05:
                ax.text(x_baseline + safe_w / 2, y_center, f"{safe_prob:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)
                
        # Hole part (black)
        if hole_w > 0:
            rect_hole = patches.Rectangle((x_baseline + safe_w, y_bottom), hole_w, height, facecolor="black", edgecolor="black", alpha=0.9)
            ax.add_patch(rect_hole)
            if hole_prob > 0.05:
                ax.text(x_baseline + safe_w + hole_w / 2, y_center, f"{hole_prob:.2f}", ha='center', va='center', color='white', fontweight='bold', fontsize=8)

def draw_spatial_consequences_subplot(ax, safe_probs, hole_probs, unexplored, title):
    # Set limits and clean axes
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    # 1. Draw Center Cell (State 10) - matches background cell size exactly
    rect_center = patches.Rectangle((0.5, 0.5), 1.0, 1.0, facecolor='#eeeeee', edgecolor='black', linewidth=1.5)
    ax.add_patch(rect_center)
    ax.text(1.0, 1.0, "State 10", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    
    # 2. Draw Stacked Bars for Actions (directly attached, thickness reduced to 0.3)
    # Up: action 3 (horizontal bar, [0.5, 1.5] x [1.5, 1.8])
    draw_stacked_horizontal_bar(ax, 0.5, 1.65, 1.0, 0.3, safe_probs[3], hole_probs[3], unexplored[3])
    ax.text(1.0, 1.85, "Up", ha='center', va='bottom', fontsize=9, fontweight='semibold', color='black')
    
    # Down: action 1 (horizontal bar, [0.5, 1.5] x [0.2, 0.5])
    draw_stacked_horizontal_bar(ax, 0.5, 0.35, 1.0, 0.3, safe_probs[1], hole_probs[1], unexplored[1])
    ax.text(1.0, 0.15, "Down", ha='center', va='top', fontsize=9, fontweight='semibold', color='black')
    
    # Left: action 0 (vertical bar, [0.2, 0.5] x [0.5, 1.5])
    draw_stacked_vertical_bar(ax, 0.35, 0.5, 0.3, 1.0, safe_probs[0], hole_probs[0], unexplored[0])
    ax.text(0.15, 1.0, "Left", ha='right', va='center', fontsize=9, fontweight='semibold', color='black')
    
    # Right: action 2 (vertical bar, [1.5, 1.8] x [0.5, 1.5])
    draw_stacked_vertical_bar(ax, 1.65, 0.5, 0.3, 1.0, safe_probs[2], hole_probs[2], unexplored[2])
    ax.text(1.85, 1.0, "Right", ha='left', va='center', fontsize=9, fontweight='semibold', color='black')

def plot_consequences_for_seed(snapshots, checkpoints, first_goal_episode, seed, save_path):
    target_state = 10
    ground_truth = get_ground_truth()
    
    # 3x4 Grid layout:
    # Subplot 0: Ground Truth
    # Subplots 1-11: Checkpoints
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=300)
    axes = axes.flatten()
    
    # 1. Plot Ground Truth in first slot
    gt_safe = [ground_truth[a]["Safe"] for a in range(4)]
    gt_hole = [ground_truth[a]["Hole"] for a in range(4)]
    gt_unexplored = [False] * 4
    draw_spatial_consequences_subplot(axes[0], gt_safe, gt_hole, gt_unexplored, "Ground Truth (True Safety)")
            
    # 2. Plot checkpoints
    for i, ep in enumerate(checkpoints, start=1):
        if i >= len(axes):
            break
        ax = axes[i]
        _, _, _, model = snapshots[ep]
        
        ch_safe = []
        ch_hole = []
        unexplored = []
        
        for a in range(4):
            counts = model.get(target_state, {}).get(a, {})
            total = sum(counts.values())
            
            if total == 0:
                ch_safe.append(0.0)
                ch_hole.append(0.0)
                unexplored.append(True)
            else:
                hole_count = sum(counts.get(h, 0) for h in HOLES)
                safe_count = total - hole_count
                ch_safe.append(safe_count / total)
                ch_hole.append(hole_count / total)
                unexplored.append(False)
                
        title_suffix = ""
        if ep == first_goal_episode:
            title_suffix = " (First Goal)"
        draw_spatial_consequences_subplot(ax, ch_safe, ch_hole, unexplored, f"Episode {ep}{title_suffix}")
        
    # Create custom legend patches
    safe_patch = patches.Patch(facecolor='forestgreen', edgecolor='black', label='Safe State', alpha=0.9)
    hole_patch = patches.Patch(facecolor='black', edgecolor='black', label='Hole (Failure)', alpha=0.9)
    unexp_patch = patches.Patch(facecolor='lightgray', hatch='//', edgecolor='gray', label='Unexplored (N/A)', alpha=0.6)
    
    # Place figure-level legend at the bottom right of the figure
    fig.legend(handles=[safe_patch, hole_patch, unexp_patch], loc="lower right", bbox_to_anchor=(0.98, 0.015), ncol=3, frameon=True, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Action consequences plot saved successfully to: {save_path}")
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
            
            # Select 3x4 display subset for action consequences plot
            action_checkpoints = {1, 5, 10, 20, 30, 40, 50, 100, 250, 1000}
            if first_goal_episode is not None:
                action_checkpoints.add(first_goal_episode)
            action_checkpoints = sorted(list(action_checkpoints))
            
            save_path = f"results/action_consequences_state10_seed{seed}.png"
            plot_consequences_for_seed(snapshots, action_checkpoints, first_goal_episode, seed, save_path)
        else:
            print(f"Warning: Seed {seed} not found in cache data.")

if __name__ == "__main__":
    main()
