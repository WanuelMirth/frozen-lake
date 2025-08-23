# agents/uct_rbql_agent.py
import numpy as np

class UCTRBQLAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.discount_rate = params["discount_rate"]
        
        # --- Q-TABLE UND MODELL BLEIBEN GLEICH ---
        self.q_table = {} 
        self.model = {}
        self.rewards = {}
        
        # --- NEUE UCT-KOMPONENTEN ---
        # Ersetzen Epsilon durch Zähler und eine Explorationskonstante
        self.n_s = {}  # Zählt Besuche des Zustands s: N(s)
        self.n_sa = {} # Zählt Ausführungen der Aktion a in Zustand s: N(s,a)
        # c ist ein Hyperparameter, der die Neugier des Agenten steuert
        self.exploration_constant_c = params.get("exploration_constant_c", 1.4) 
        
        print(f"UCT-RBQL Agent initialisiert mit c={self.exploration_constant_c}.")

    def _get_q_values(self, state):
        return self.q_table.get(state, np.zeros(self.n_actions))

    def choose_action(self, state):
        # Wenn ein Zustand komplett neu ist, initialisiere Zähler und exploriere zufällig
        if state not in self.n_s:
            self.n_s[state] = 0
            self.n_sa[state] = np.zeros(self.n_actions)
            return self.action_space.sample()

        uct_values = np.zeros(self.n_actions)
        q_values = self._get_q_values(state)

        for action in range(self.n_actions):
            # Wenn eine Aktion noch nie ausgeführt wurde, wähle sie sofort aus.
            # Das ist entscheidend, um sicherzustellen, dass alles erkundet wird.
            if self.n_sa[state][action] == 0:
                return action
            
            # 1. Exploitation-Term: Nutze das Wissen aus der Q-Tabelle (gelernt durch RBQL)
            exploitation_term = q_values[action]
            
            # 2. Exploration-Term: Sei neugierig auf seltener genutzte Aktionen
            exploration_term = self.exploration_constant_c * \
                np.sqrt(np.log(self.n_s[state]) / self.n_sa[state][action])
            
            uct_values[action] = exploitation_term + exploration_term
            
        # Wähle die Aktion mit dem höchsten UCT-Wert
        return np.argmax(uct_values)

    def learn(self, state, action, reward, new_state):
        # --- MODELL-UPDATE (bleibt unverändert) ---
        if state not in self.model:
            self.model[state] = {a: {} for a in range(self.n_actions)}
            self.rewards[state] = {a: {} for a in range(self.n_actions)}
        
        outcomes = self.model[state][action]
        outcomes[new_state] = outcomes.get(new_state, 0) + 1
        
        if new_state not in self.rewards[state][action]:
            self.rewards[state][action][new_state] = []
        self.rewards[state][action][new_state].append(reward)

        # --- ZÄHLER-UPDATE FÜR UCT ---
        self.n_s[state] = self.n_s.get(state, 0) + 1
        if state not in self.n_sa:
             self.n_sa[state] = np.zeros(self.n_actions)
        self.n_sa[state][action] += 1

    def on_episode_end(self, episode, episode_reward):
        # Die Logik wird radikal vereinfacht: Nur noch das Backwards-Lernen ausführen.
        # Kein Epsilon-Decay oder Zielfindungs-Flag mehr nötig.
        self._learn_backwards()

    def _learn_backwards(self):
        all_known_states = set(self.model.keys())
        
        # Iteriere oft genug, um Konvergenz zu gewährleisten
        for _ in range(len(all_known_states) * 2): 
            for state in self.model:
                # WICHTIGER FIX: Iteriere über ALLE Aktionen, nicht nur die besuchten
                for action in range(self.n_actions):
                    
                    if action not in self.model[state] or not self.model[state][action]:
                        continue # Diese Aktion wurde von diesem Zustand aus noch nie ausgeführt

                    outcomes = self.model[state][action]
                    total_transitions = sum(outcomes.values())
                    if total_transitions == 0:
                        continue

                    expected_q_value = 0
                    for next_state, count in outcomes.items():
                        probability = count / total_transitions
                        avg_reward = np.mean(self.rewards[state][action][next_state])
                        next_max = np.max(self._get_q_values(next_state))
                        branch_value = probability * (avg_reward + self.discount_rate * next_max)
                        expected_q_value += branch_value
                    
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(self.n_actions)
                    self.q_table[state][action] = expected_q_value