import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def generate_smooth_plot(csv_path, window_size=500):
    """
    Liest eine aggregierte CSV-Datei, glättet die mittlere Belohnung und erstellt
    einen visuell aussagekräftigen Vergleich, der die schnelle Konvergenz des
    UCT-RBQL-Agenten hervorhebt.
    """
    if not os.path.exists(csv_path):
        print(f"Fehler: Datei nicht gefunden unter '{csv_path}'")
        return

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("Metriken-Datei ist leer, kein Plot wird erstellt.")
            return

        # Berechne den gleitenden Durchschnitt
        df['smoothed_mean'] = df['mean'].rolling(window=window_size, min_periods=1).mean()

        # Berechne das Konvergenzniveau des UCT-RBQL Agenten
        if len(df) > 10:
            num_last_episodes = int(len(df) * 0.1)
            if num_last_episodes < window_size:
                num_last_episodes = window_size
            convergence_level = df['smoothed_mean'].iloc[-num_last_episodes:].mean()
        else:
            convergence_level = df['smoothed_mean'].iloc[-1]

        # Erstelle den Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        # --- KORRIGIERTE LEGENDEN-EINTRÄGE ---
        uct_rbql_color = 'firebrick'
        optimized_mcts_color = 'royalblue'

        # Plot des UCT-RBQL Agenten
        ax.plot(df['episode'], df['smoothed_mean'], color=uct_rbql_color, linewidth=2.5, label='UCT-RBQL')
        
        # Horizontale Linie für das Konvergenzniveau von UCT-RBQL
        ax.axhline(y=convergence_level, color=uct_rbql_color, linestyle='--', linewidth=1.5)

        # Horizontale Linie für die Performance von Optimized MCTS
        ax.axhline(y=0.75, color=uct_rbql_color, linestyle='--', linewidth=1.5, label='Konvergenzniveau UCT-RBQL')

        # Horizontale Linie für die Performance von Optimized MCTS
        ax.axhline(y=0.75, color=optimized_mcts_color, linestyle='--', linewidth=1.5, label='Konvergenzniveau Optimized MCTS')



        # -----------------------------------------------------------------
        
        ax.set_title("UCT-RBQL", fontsize=16, pad=20)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Kumulative Belohnung (geglättet)", fontsize=12)
        
        # Dynamische Y-Achsen-Skalierung
        max_val = df['smoothed_mean'].max()
        ax.set_ylim(-0.05, max_val * 1.05 + 0.05)

        # Text für Konvergenzniveaus an der Y-Achse
        ax.text(-0.02, convergence_level, f'{convergence_level:.2f}', 
                va='center', ha='right', transform=ax.get_yaxis_transform(), 
                fontsize=10, color=uct_rbql_color)
        ax.text(-0.02, 0.75, '0.75', 
                va='center', ha='right', transform=ax.get_yaxis_transform(), 
                fontsize=10, color=optimized_mcts_color)
        
        ax.legend(loc='lower right')
        fig.tight_layout()
        
        new_plot_path = csv_path.replace('.csv', '_comparison_plot3.png')
        fig.savefig(new_plot_path, bbox_inches='tight')
        print(f"\nVergleichs-Plot der Ergebnisse gespeichert unter: {new_plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erstellt einen geglätteten Plot aus aggregierten Ergebnis-CSVs.")
    parser.add_argument("csv_path", type=str, help="Der Pfad zur aggregierten CSV-Datei.")
    parser.add_argument("--window", type=int, default=200, help="Fenstergröße für den gleitenden Durchschnitt.")
    
    args = parser.parse_args()
    
    generate_smooth_plot(args.csv_path, args.window)