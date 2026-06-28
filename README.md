
***

# QUEST Agent on Slippery FrozenLake-v1

**Q**-learning via **U**CB **E**xploration and **S**ystematic **T**erminal-sweeps

---

## 1. The Environment: Stochastic FrozenLake-v1

Evaluated on the highly stochastic `is_slippery=True` gymnasium environments:
*   **4x4 Grid**: 16 discrete states 
*   **8x8 Grid**: 64 discrete states
*   **Dynamics**: The agent only moves in the intended direction with a $\frac{1}{3}$ probability. There is a $\frac{2}{3}$ chance of slipping orthogonally. Because of this noise, an optimal oracle policy caps out at approximately $77.8\%$ on the 4x4 grid.

---

## 2. The Algorithm (QUEST)

QUEST is a Model-Based Reinforcement Learning architecture designed to overcome the severe overestimation bias that plagues traditional backward-planning and deep exploration methods in highly noisy MDPs. 

It completely decouples value estimation from exploration by:
1. Building an objective, empirical transition and reward model ($\hat{P}$, $\hat{R}$).
2. Disabling mid-episode updates, instead deferring value propagation to an exact, offline Value Iteration sweep upon episode termination.
3. Utilizing a local, count-based Upper Confidence Bound (UCB) strictly during online action selection to keep the underlying Bellman equations uncorrupted by exploration optimism.

---

## 3. Multi-Objective HPO (Optuna)

Hyperparameter Optimization (HPO) was performed over 100 Trials (using 10 seeds for 4x4, and 5 seeds for 8x8) using **Optuna** to optimize two conflicting objectives:
1.  **Minimize Convergence Time**
2.  **Maximize Final Performance**

To view the Optuna dashboard locally:
```bash
# For 4x4 Grid
uvx optuna-dashboard QUEST_Multi_Objective_4x4.log

# For 8x8 Grid
uvx optuna-dashboard QUEST_Multi_Objective_8x8.log
```

---

## 4. Results (100-Seed Large-Scale Validation)

The optimal configurations discovered by Optuna were rigorously validated across a massive ensemble of **100 independent seeds** to establish strict, noise-resistant asymptotic performance floors.

| Environment | Optimal Run | Exploration $c$ | Discount $\gamma$ | Convergence Ep | Asymptotic Success Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **4x4 Grid** | **Trial 51** | $0.0907$ | $0.9917$ | **353** (to $\ge 70\%$) | **73.02%** |
| **8x8 Grid** | **Trial 99** | $0.0138$ | $0.9988$ | **1,478** (to $\ge 85\%$)| **88.94%** |

QUEST achieves state-of-the-art sample efficiency, outpacing traditional planning algorithms like Optimized MCTS by orders of magnitude and matching the rapid convergence of modern large-scale in-context models (like OmniRL) while providing a much more stable performance limit.

---

## 5. Installation & Dependency Versions

To ensure deterministic environments, dependencies are locked in `uv.lock`.

### Syncing Locked Versions
To create a local virtual environment and install the exact package versions used to generate these results, run:
```bash
uv sync
```
This reads the locked versions from `uv.lock` (including `gymnasium==1.3.0`, `optuna==4.8.0`, etc.) and configures your virtual environment.

### Running scripts inside the uv environment
*   **Run parallel seed sweeps**:
    ```bash
    uv run python run_4x4_100seeds_sweep.py
    ```
