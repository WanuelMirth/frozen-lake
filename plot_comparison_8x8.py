# plot_comparison_8x8.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_comparison_plot():
    # Compare optimal QUEST 8x8 (Trial 99) with our R-Max 8x8 config
    quest_csv = 'results/8x8_sweep_100seeds/AGGREGATED_QUEST_Pareto_Trial99.csv'
    rmax_csv = 'results/8x8_sweep_100seeds/AGGREGATED_RMax_Best_8x8_Config.csv'
    
    if not os.path.exists(quest_csv) or not os.path.exists(rmax_csv):
        print(f"Error: Aggregated 8x8 CSV files not found. Check if both sweeps have run.")
        return
        
    df_quest = pd.read_csv(quest_csv)
    df_rmax = pd.read_csv(rmax_csv)
    
    window_size = 100
    df_quest['smoothed_mean'] = df_quest['mean'].rolling(window=window_size, min_periods=1).mean()
    df_rmax['smoothed_mean'] = df_rmax['mean'].rolling(window=window_size, min_periods=1).mean()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    quest_perf = df_quest['smoothed_mean'].iloc[-1000:].mean()
    rmax_perf = df_rmax['smoothed_mean'].iloc[-1000:].mean()
    
    # Colors
    color_quest = 'crimson'
    color_rmax = 'royalblue'
    
    # Plot curves
    ax.plot(df_quest['episode'], df_quest['smoothed_mean'], color=color_quest, linewidth=2, 
            label=f'QUEST (Trial 99) | Final Perf: {quest_perf*100:.2f}%')
    ax.plot(df_rmax['episode'], df_rmax['smoothed_mean'], color=color_rmax, linewidth=2, 
            label=f'R-Max (Best 8x8 Config) | Final Perf: {rmax_perf*100:.2f}%')
            
    # Find convergence episodes (rolling 100-ep mean >= 85% for 8x8)
    threshold = 0.85
    
    quest_cross = df_quest[df_quest['smoothed_mean'] >= threshold].index
    quest_conv = int(df_quest.loc[quest_cross[0], 'episode']) if len(quest_cross) > 0 else None
    
    rmax_cross = df_rmax[df_rmax['smoothed_mean'] >= threshold].index
    rmax_conv = int(df_rmax.loc[rmax_cross[0], 'episode']) if len(rmax_cross) > 0 else None
    
    # Plot convergence vertical lines
    if rmax_conv:
        ax.axvline(x=rmax_conv, color=color_rmax, linestyle='--', linewidth=1.5, alpha=0.8, 
                   label=f'R-Max Convergence (85%): Ep {rmax_conv}')
    if quest_conv:
        ax.axvline(x=quest_conv, color=color_quest, linestyle='--', linewidth=1.5, alpha=0.8, 
                   label=f'QUEST Convergence (85%): Ep {quest_conv}')
                   
    # Compute non-overlapping ticks for convergence episodes
    base_ticks = [0, 1000, 2000, 3000, 4000, 5000]
    final_ticks = list(base_ticks)
    
    for t in [rmax_conv, quest_conv]:
        if t is not None:
            # Only add to x-axis ticks if it's not too close (within 100 episodes) to other ticks
            if all(abs(t - existing) > 100 for existing in final_ticks):
                final_ticks.append(t)
            
    ax.set_xticks(sorted(final_ticks))
    
    ax.set_title("Learning Curve Comparison (8x8): QUEST vs R-Max (100 Seeds)", fontsize=14, pad=15)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Success Rate (Smoothed)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    fig.tight_layout()
    
    output_path = 'results/QUEST_vs_RMax_comparison_8x8.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Comparison plot successfully saved to: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    generate_comparison_plot()
