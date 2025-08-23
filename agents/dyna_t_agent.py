# agents/dyna_t_agent.py
import numpy as np
import random

class DynaTAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.q_table = np.zeros((observation_space.n, self.n_actions))
        
        # --- KORRIGIERTES, PROBABILISTISCHES MODELL ---
        self.model = {}      # model[s][a] -> {s': count}
        self.rewards = {}    # rewards[s][a] -> {s': [list_of_rewards]}
        self.planning_steps = params.get("planning_steps", 50)

        # --- UCT ZÄHLER FÜR AKTIONSAUSWAHL ---
        self.ns = np.zeros(observation_space.n, dtype=int)
        self.nsa = np.zeros((observation_space.n, self.n_actions), dtype=int)
        self.exploration_constant_c = params.get("exploration_constant_c", 2.0)

        self.learning_rate = params["learning_rate"]
        self.discount_rate = params["discount_rate"]
        
        print(f"Finaler Stochastic Dyna-T Agent initialisiert (UCT-Policy, probabilistisches Modell).")

    def choose_action(self, state):
        # UCT-basierte Aktionsauswahl für intelligente Exploration
        uct_values = np.zeros(self.n_actions)
        for action in range(self.n_actions):
            q_value = self.q_table[state, action]
            if self.nsa[state, action] == 0:
                exploration_bonus = float('inf')
            else:
                exploration_bonus = self.exploration_constant_c * \
                    np.sqrt(np.log(self.ns[state] + 1) / self.nsa[state, action])
            uct_values[action] = q_value + exploration_bonus
        return np.argmax(uct_values)

    def learn(self, state, action, reward, new_state):
        # Update der UCT-Zähler mit der echten Erfahrung
        self.ns[state] += 1
        self.nsa[state, action] += 1

        # 1. Direktes Lernen von der echten Erfahrung
        self._q_update(state, action, reward, new_state)
        
        # 2. PROBABILISTISCHES MODELL-UPDATE
        if state not in self.model:
            self.model[state] = {a: {} for a in range(self.n_actions)}
            self.rewards[state] = {a: {} for a in range(self.n_actions)}
        
        outcomes = self.model[state][action]
        outcomes[new_state] = outcomes.get(new_state, 0) + 1
        
        if new_state not in self.rewards[state][action]:
            self.rewards[state][action][new_state] = []
        self.rewards[state][action][new_state].append(reward)
        
        # 3. PLANUNGSPHASE MIT KORREKTEM MODELL
        for _ in range(self.planning_steps):
            if not self.model: break
            
            s_rand = random.choice(list(self.model.keys()))
            if not self.model[s_rand]: continue # Falls von s_rand noch keine Aktionen bekannt
            a_rand = random.choice(list(self.model[s_rand].keys()))
            if not self.model[s_rand][a_rand]: continue # Falls für s_rand, a_rand noch kein Ausgang bekannt
            
            # --- SAMPLING AUS DEM PROBABILISTISCHEN MODELL ---
            outcomes = self.model[s_rand][a_rand]
            possible_next_states = list(outcomes.keys())
            counts = list(outcomes.values())
            total = sum(counts)
            probabilities = [c / total for c in counts]
            
            # Wähle einen "geträumten" Ausgang basierend auf den gelernten Wahrscheinlichkeiten
            s_prime_model = np.random.choice(possible_next_states, p=probabilities)
            
            # Hole die durchschnittliche Belohnung für diesen geträumten Übergang
            r_model = np.mean(self.rewards[s_rand][a_rand][s_prime_model])
            
            self._q_update(s_rand, a_rand, r_model, s_prime_model)

    def _q_update(self, state, action, reward, new_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[new_state, :])
        new_value = old_value + self.learning_rate * (reward + self.discount_rate * next_max - old_value)
        self.q_table[state, action] = new_value

    def on_episode_end(self, episode, episode_reward):
        # Diese Methode wird nicht für Epsilon-Decay benötigt
        pass