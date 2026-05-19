## 1. Context & Insights from Phase 1 (Baseline)

* **Environment:** `FrozenLake-v1` with `is_slippery=True`. Due to stochastic slipping, the theoretical maximum reward even for a perfect agent is capped around **0.78 - 0.82**.
* **Agent:** `UCT-RBQL` (Monte Carlo Tree Search). This algorithm is purely **CPU-bound** (heavy logical branching, no neural networks). A GPU is *not* required.
* **V1 Insights:** The agent successfully reaches solid scores of **~0.7400** but converges incredibly early. Running 5,000 episodes is wasting computational cycles.
* **Fixes implemented in V2:**
  1. *Anti-Overfitting:* Switched away from the pre-experiment seed (42) to a robust multi-seed setup.
  2. *Multi-Objective:* Moving away from single-metric evaluation to optimize both end-performance and learning speed simultaneously.
  3. *Precision Tracking:* Forces all metrics and evaluation scores to keep a precision of **4 decimal places** (`.4f`) for fine-grained analysis.
  4. *Thread-Safety:* Completely disabled local Matplotlib rendering to prevent Tkinter parallel worker crashes. Intermediate values are saved directly to the database.

---

## 2. Phase 1 Bootstrapping (Warmstart Data)

We have completed a baseline 100-trial single-objective run using **Seed 42**. To speed up the new Multi-Objective run, we inject the best parameters from Phase 1 as the initial trials.
--- Optimization Finished ---
Best trial: 21
  Value: 0.74
  Params: {'exploration_constant_c': 0.10503610290517224, 'discount_rate': 0.9951260597133066}

--- Top 5 Trials ---
 number  value  params_exploration_constant_c  params_discount_rate
     62   0.74                       0.071077              0.985246
     22   0.74                       0.086279              0.969303
     85   0.74                       0.041012              0.994155
     84   0.74                       0.038317              0.993550
     83   0.74                       0.091211              0.992617

All trial results saved to: optuna_results_UCT_RBQL_FrozenLake_Optimization_20260518_192832.csv


---

## 3. Target Architecture for Version 2 (The Next Gen)

Designed to run natively on a **MacBook M4 Pro** via `uv`, wrapped in a `caffeinate` session to bypass corporate/IT sleep policies.

### Core Features:
1. **Multi-Seed Evaluation:** Every single Optuna trial evaluates its parameters across **3 entirely new environments** (`seeds = [111, 222, 333]`) and optimizes the average performance to guarantee generalizability.
2. **Multi-Objective Optimization (Pareto Front):** Optuna tracks two distinct goals:
   * **Objective 0 (Maximize):** The final score (mean reward of the last 500 episodes - tracked and stored with **4 decimal places**).
   * **Objective 1 (Minimize):** Convergence speed (the episode index where a 100-episode rolling average reward of **0.7000** is cracked permanently).
3. **Efficiency:** Maximum training episodes cut down from 5,000 to **2,000**.
4. **Intermediate Tracking:** `trial.report()` saves training histories directly to the SQLite database every 100 episodes, populating the **Intermediate Values** tab in the dashboard with live, interactive learning curves.

---

## 4. Code Specifications

### A. Modifications inside `main.py`
The `train` function must accept the optional `trial` parameter, drop any local plotting calls, keep 4 decimal places for logging, and report its progress periodically.
CRITICAL: Comment out or remove any Matplotlib GUI plotting at the end!
Inside the training loop, report intermediate values every 100 episodes:

## 5. Execution Workflow
Terminal Tab 1: Run optimization while forcing the system to stay awake
caffeinate -dims uv run optuna_multi_objective.py

Terminal Tab 2: Launch the dashboard server to observe the live Pareto Front & Contours
uvx optuna-dashboard sqlite:///UCT_RBQL_Multi_Objective.db