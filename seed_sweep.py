# seed_sweep.py
import os
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import main # Wir importieren die train-Funktion

# --- 1. DEFINIERE DIE CHAMPION-KONFIGURATIONEN ---
# Hier trägst du die besten Hyperparameter ein, die du für jeden Agenten gefunden hast.
CHAMPION_CONFIGS = {
    "Dyna_T_4x4_Slippery_CHAMPION": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "Dyna-T",
        "total_episodes": 5000, "max_steps_per_episode": 200,
        "learning_rate": 0.05, "planning_steps": 50, "exploration_constant_c": 0.2,
        "discount_rate": 0.99, "render": False,
    },
    "Q_Learning_4x4_Slippery_CHAMPION": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "Q-Learning",
        "total_episodes": 10000, "max_steps_per_episode": 200,
        "learning_rate": 0.05, "discount_rate": 0.99,
        "max_epsilon": 1.0, "min_epsilon": 0.01, "epsilon_decay_rate": 0.0002,
        "render": False,
    },
    # NEU: Der Champion für StochasticRBQL, basierend auf Ihren Sweep-Ergebnissen
    "StochasticRBQL_4x4_Slippery_CHAMPION": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "StochasticRBQL",
        "total_episodes": 5000, "max_steps_per_episode": 200,
        "discount_rate": 0.99, "max_epsilon": 1.0, "min_epsilon": 0.05,
        "epsilon_decay_rate": 0.001, # Der Gewinner Ihres Sweeps
        "render": False,
    },
    "UCT_RBQL_4x4_Slippery": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "agent": "UCT-RBQL",  # Wähle den neuen Agenten
        "total_episodes": 5000, "max_steps_per_episode": 200,
        "discount_rate": 0.99,
        "exploration_constant_c": 0.1, # Der neue, wichtige Hyperparameter!
        "render": False,
    },
}

# --- 2. DEFINIERE DIE PARAMETER DES SWEEPS ---
NUM_SEEDS = 15 # Anzahl der Durchläufe pro Konfiguration

# --- 3. FUNKTION ZUM DURCHFÜHREN DES SEED-SWEEPS ---
def run_seed_sweep(config_name):
    try:
        base_config = CHAMPION_CONFIGS[config_name]
    except KeyError:
        print(f"Fehler: Champion-Konfiguration '{config_name}' nicht gefunden. Verfügbar: {list(CHAMPION_CONFIGS.keys())}")
        return []
        
    all_run_dirs = []
    
    print(f"\n===== STARTE SEED-SWEEP FÜR: {config_name} ({NUM_SEEDS} DURCHLÄUFE) =====")
    
    for seed in range(NUM_SEEDS):
        config = base_config.copy()
        config['seed'] = seed # Füge den Seed zur Konfiguration hinzu
        
        # Erstelle einen eindeutigen Ordnernamen für diesen spezifischen Lauf
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config_name}_seed{seed}_{timestamp}"
        
        print(f"\n--- Starte Lauf mit Seed {seed}/{NUM_SEEDS-1} ---")
        results_path, _ = main.train(run_name, config)
        all_run_dirs.append(results_path)
        
    return all_run_dirs

# --- 4. FUNKTION ZUR AGGREGIERUNG UND VISUALISIERUNG ---
def aggregate_and_plot(all_dirs, base_run_name):
    all_dfs = []
    for run_dir in all_dirs:
        try:
            df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warnung: metrics.csv nicht gefunden in {run_dir}")

    if not all_dfs:
        print("Keine Daten zum Aggregieren gefunden.")
        return

    combined_df = pd.concat(all_dfs)
    aggregated_df = combined_df.groupby('episode')['reward'].agg(['mean', 'std']).reset_index()
    aggregated_df['std'] = aggregated_df['std'].fillna(0)
    
    summary_path = os.path.join("results", f"AGGREGATED_{base_run_name}.csv")
    aggregated_df.to_csv(summary_path, index=False)
    print(f"\nAggregierte Ergebnisse gespeichert unter: {summary_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    mean = aggregated_df['mean']
    std = aggregated_df['std']
    episodes = aggregated_df['episode']
    
    ax.plot(episodes, mean, label='Durchschnittliche Belohnung')
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.2, label='Standardabweichung')
    
    ax.set_title(f"Aggregierter Trainingsverlauf für {base_run_name} ({NUM_SEEDS} Seeds)", fontsize=16)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Kumulative Belohnung", fontsize=12)
    ax.legend()
    ax.set_ylim(-0.05, 1.05) # Sorge für eine konsistente Y-Achse
    
    plot_path = os.path.join("results", f"AGGREGATED_{base_run_name}_plot.png")
    fig.savefig(plot_path)
    print(f"Aggregierter Plot gespeichert unter: {plot_path}")
    plt.close(fig)

if __name__ == "__main__":
    # --- WÄHLE HIER AUS, WELCHEN CHAMPION DU TESTEN MÖCHTEST ---
    config_to_sweep = "UCT_RBQL_4x4_Slippery"
    # config_to_sweep = "Dyna_T_4x4_Slippery_CHAMPION"
    # config_to_sweep = "Q_Learning_4x4_Slippery_CHAMPION"
    
    run_directories = run_seed_sweep(config_to_sweep)
    if run_directories:
        aggregate_and_plot(run_directories, config_to_sweep)