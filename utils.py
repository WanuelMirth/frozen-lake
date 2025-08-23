# utils.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_dir, config):
    """Liest die Metriken und erstellt einen Plot, der im Ergebnisordner gespeichert wird."""
    try:
        df = pd.read_csv(f"{results_dir}/metrics.csv")
        if df.empty:
            print("Metriken-Datei ist leer, kein Plot wird erstellt.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot der rohen Belohnungen pro Episode
        ax.plot(df['episode'], df['reward'], 'o', color='lightgray', markersize=2, label='Belohnung pro Episode')

        # Plot eines gleitenden Durchschnitts für den Trend
        window_size = 50
        moving_avg = df['reward'].rolling(window=window_size, min_periods=1).mean()
        ax.plot(df['episode'], moving_avg, color='steelblue', linewidth=2, label=f'Gleitender Durchschnitt (Fenster={window_size})')
        
        ax.set_title(f"Trainingsverlauf für Agent: {config['agent']} auf {config['env_name']}-{config['map_name']}", fontsize=16)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Kumulative Belohnung", fontsize=12)
        ax.legend(loc='lower right')
        
        fig.tight_layout()
        plot_path = f"{results_dir}/training_plot.png"
        fig.savefig(plot_path)
        print(f"Plot der Ergebnisse gespeichert unter: {plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Fehler beim Erstellen des Plots: {e}")