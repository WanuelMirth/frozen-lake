import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevent Tkinter parallel/worker GUI crash
import matplotlib.pyplot as plt
import argparse
import os
import glob

def generate_individual_plot(csv_path, window_size=100):
    """
    Generates a high-quality learning curve plot for a single sweep CSV,
    showing the smoothed mean over 25 seeds.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found at '{csv_path}'")
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV {csv_path} is empty.")
            return None

        # Smoothing
        df['smoothed_mean'] = df['mean'].rolling(window=window_size, min_periods=1).mean()

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        is_target_run = 'Pareto_Trial74' in csv_path or 'Pareto_Trial92' in csv_path

        final_perf = df['smoothed_mean'].iloc[-1000:].mean()

        if 'Pareto_Trial74' in csv_path:
            color = 'teal'
            label = f'Success Rate (Avg Last 1000 Ep: {final_perf:.4f})'
        elif 'Pareto_Trial92' in csv_path:
            color = 'crimson'
            label = f'Success Rate (Avg Last 1000 Ep: {final_perf:.4f})'
        else:
            color = 'royalblue'
            label = os.path.basename(csv_path).replace('AGGREGATED_', '').replace('.csv', '').replace('_', ' ')

        # Plot mean line (no std dev)
        line_label = label if is_target_run else f'{label} (Mean)'
        ax.plot(df['episode'], df['smoothed_mean'], color=color, linewidth=2, label=line_label)



        if is_target_run:
            title_label = label
        else:
            title_label = label + " (25 Seeds)"
        ax.set_title(f"Learning Curve: {title_label}", fontsize=14, pad=15)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Reward (Smoothed)", fontsize=11)
        ax.set_ylim(-0.05, 1.05)

        if is_target_run:
            # Mark the convergence episode on x axis
            cross_indices = df[df['smoothed_mean'] >= 0.7].index
            conv_ep = int(df.loc[cross_indices[0], 'episode']) if len(cross_indices) > 0 else 254
            ax.axvline(x=conv_ep, color='dimgray', linestyle=':', alpha=0.8, label=f'Convergence: Ep {conv_ep}')
            # Update xticks to include conv_ep and prevent overlapping text
            current_ticks = [0, 1000, 2000, 3000, 4000, 5000]
            new_ticks = [t for t in current_ticks if abs(t - conv_ep) > 250]
            new_ticks.append(conv_ep)
            ax.set_xticks(sorted(new_ticks))

        ax.legend(loc='lower right', frameon=True)
        fig.tight_layout()

        plot_path = csv_path.replace('.csv', '_smooth_plot.png')
        fig.savefig(plot_path, bbox_inches='tight')
        print(f"Individual plot saved to: {plot_path}")
        plt.close(fig)
        return df

    except Exception as e:
        print(f"An error occurred plotting {csv_path}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate smoothed plots from aggregated seed sweeps.")
    parser.add_argument("--csv_path", type=str, default="", help="Optional single CSV to plot.")
    parser.add_argument("--window", type=int, default=100, help="Smoothing window size.")
    args = parser.parse_args()

    results_dir = "results"

    if args.csv_path:
        generate_individual_plot(args.csv_path, args.window)
    else:
        # Auto-detect all AGGREGATED_*.csv files recursively in results/
        csv_files = glob.glob(os.path.join(results_dir, "**/AGGREGATED_*.csv"), recursive=True)
        for csv_file in csv_files:
            generate_individual_plot(csv_file, args.window)
            # Copy to results root directory for visibility
            filename = os.path.basename(csv_file).replace('.csv', '_smooth_plot.png')
            dest = os.path.join(results_dir, filename)
            src = csv_file.replace('.csv', '_smooth_plot.png')
            if os.path.exists(src):
                import shutil
                shutil.copy(src, dest)