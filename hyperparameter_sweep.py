# hyperparameter_sweep.py
import itertools
import os
import pandas as pd
from datetime import datetime
import json
import main # Wir importieren die train-Funktion

# --- 1. DEFINIERE DIE SWEEP-KONFIGURATION ---
# Wähle hier aus, welchen Agenten du sweepen möchtest
AGENT_TO_SWEEP = "UCT-RBQL" # Fokus auf den UCT-RBQL Agenten

# --- NEU: Liste von Seeds für Robustheits-Check ---
SEEDS = [43]

# --- 2. ERWEITERTE SUCHRÄUME FÜR JEDEN AGENTEN ---
SEARCH_SPACES = {
    "Q-Learning": {
        'learning_rate': [0.1, 0.2],
        'discount_rate': [0.98, 0.99],
        'epsilon_decay_rate': [0.001, 0.0005]
    },
    "Dyna-Q": {
        'learning_rate': [0.1, 0.2, 0.4],
        'planning_steps': [25, 50],
        'discount_rate': [0.98, 0.99]
    },
    "Dyna-T": {
        'learning_rate': [0.1, 0.2],
        'planning_steps': [25, 50],
        'exploration_constant_c': [0.5, 1.0],
        'discount_rate': [0.98, 0.99]
    },
    "StochasticRBQL": {
        'epsilon_decay_rate': [0.0008, 0.001, 0.0012],
        'discount_rate': [0.98, 0.99, 0.995]
    },
    # --- NEU: Angepasster Suchraum für das Feintuning ---
    "UCT-RBQL": {
        'exploration_constant_c': [0.05, 0.1, 0.15, 0.2],
        'discount_rate': [0.985, 0.99, 0.995]
    }
}

# --- 3. DEFINIERE DIE BASIS-KONFIGURATIONEN ---
BASE_CONFIGS = {
    "FrozenLake-v1_4x4_Slippery": {
        "env_name": "FrozenLake-v1", "is_slippery": True, "map_name": "4x4",
        "total_episodes": 3000, "max_steps_per_episode": 200, "render": False,
        "max_epsilon": 1.0, "min_epsilon": 0.05,
    }
}
BASE_CONFIG_TO_USE = "FrozenLake-v1_4x4_Slippery"


def run_sweep():
    search_space = SEARCH_SPACES[AGENT_TO_SWEEP]
    base_config = BASE_CONFIGS[BASE_CONFIG_TO_USE]
    base_config['agent'] = AGENT_TO_SWEEP

    keys, values = zip(*search_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_runs = len(param_combinations) * len(SEEDS)
    print(f"Starte Sweep für '{AGENT_TO_SWEEP}' mit {len(param_combinations)} Kombinationen und {len(SEEDS)} Seeds pro Lauf.")
    print(f"Insgesamt werden {total_runs} Trainingsläufe durchgeführt.")
    
    results_data = []
    run_counter = 0
    for i, params in enumerate(param_combinations):
        for seed in SEEDS:
            run_counter += 1
            config = base_config.copy()
            config.update(params)
            config['seed'] = seed
            
            run_name_suffix = "_".join([f"{key[:2]}{val}" for key, val in params.items()])
            timestamp = datetime.now().strftime("%H%M%S")
            run_name = f"Sweep_{AGENT_TO_SWEEP}_{run_name_suffix}_seed{seed}_{timestamp}"

            print(f"\n--- Starte Lauf {run_counter}/{total_runs}: {run_name} ---")
            
            results_path, duration = main.train(run_name, config)
            results_data.append((results_path, duration))
            
    return results_data

def summarize_results(results_data):
    summary_data = []
    print(f"\n--- Analysiere Ergebnisse des Sweeps für {AGENT_TO_SWEEP} ---")
    
    for results_dir, duration in results_data:
        try:
            df = pd.read_csv(os.path.join(results_dir, "metrics.csv"))
            if df.empty: continue
            config_path = os.path.join(results_dir, "config.json")
            with open(config_path) as f: config = json.load(f)
            
            final_performance = df['reward'].rolling(window=100).mean().iloc[-1]
            convergence_window = df['reward'].rolling(window=100).mean()
            convergence_episodes = None
            if convergence_window.max() >= 0.7:
                convergence_episodes = (convergence_window >= 0.7).idxmax()

            param_values = {key: config.get(key, 'N/A') for key in SEARCH_SPACES[AGENT_TO_SWEEP]}
            run_summary = {
                'run_name': os.path.basename(results_dir),
                **param_values,
                'final_performance': final_performance,
                'convergence_episodes': convergence_episodes,
                'duration_seconds': duration,
                'seed': config.get('seed', 'N/A')
            }
            summary_data.append(run_summary)
        except Exception as e:
            print(f"Konnte Ergebnis für {results_dir} nicht analysieren: {e}")

    if not summary_data:
        print("Keine Ergebnisse zum Analysieren gefunden.")
        return

    summary_df = pd.DataFrame(summary_data)
    param_keys = list(SEARCH_SPACES[AGENT_TO_SWEEP].keys())
    
    aggregated_results = summary_df.groupby(param_keys).agg(
        avg_final_performance=('final_performance', 'mean'),
        std_final_performance=('final_performance', 'std'),
        avg_convergence=('convergence_episodes', 'mean')
    ).reset_index()

    aggregated_results = aggregated_results.sort_values(by='avg_final_performance', ascending=False)
    
    print(f"\n--- AGGREGIERTE ERGEBNIS-RANGLISTE (gemittelt über {len(SEEDS)} Seeds) ---")
    print(aggregated_results.to_string())

    summary_path = os.path.join("results", f"sweep_summary_aggregated_{AGENT_TO_SWEEP}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    aggregated_results.to_csv(summary_path, index=False)
    print(f"\nZusammenfassung gespeichert unter: {summary_path}")


if __name__ == "__main__":
    all_results_data = run_sweep()
    summarize_results(all_results_data)