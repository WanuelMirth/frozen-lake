# agents/stochastic_rbql_agent.py
import numpy as np

class StochasticRBQLAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.discount_rate = params["discount_rate"]
        
        self.q_table = {} 
        
        # --- NEUES PROBABILISTISCHES MODELL ---
        # self.model[state][action] = {next_state_1: count_1, next_state_2: count_2}
        self.model = {}
        # self.rewards[state][action][next_state] = [list of rewards observed for this transition]
        self.rewards = {}
        
        self.goal_has_been_found = False
        self.epsilon = params["max_epsilon"]
        self.min_epsilon = params["min_epsilon"]
        self.max_epsilon = params["max_epsilon"]
        self.epsilon_decay_rate = params["epsilon_decay_rate"]
        
        print("Stochastic RBQL Agent initialisiert.")

    def _get_q_values(self, state):
        return self.q_table.get(state, np.zeros(self.n_actions))

    def choose_action(self, state):
        if self.goal_has_been_found and np.random.uniform(0, 1) > self.min_epsilon:
            return np.argmax(self._get_q_values(state))
        
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self._get_q_values(state))

    def learn(self, state, action, reward, new_state):
        # --- MODELL-UPDATE FÜR STOCHASTISCHE WELT ---
        # Initialisiere Dictionaries, falls nötig
        if state not in self.model:
            self.model[state] = {a: {} for a in range(self.n_actions)}
            self.rewards[state] = {a: {} for a in range(self.n_actions)}
        
        # Zähle die Häufigkeit des Ausgangs (s' | s, a)
        outcomes = self.model[state][action]
        outcomes[new_state] = outcomes.get(new_state, 0) + 1
        
        # Speichere die Belohnung für diesen spezifischen Übergang
        if new_state not in self.rewards[state][action]:
            self.rewards[state][action][new_state] = []
        self.rewards[state][action][new_state].append(reward)

    def on_episode_end(self, episode, episode_reward):
        if not self.goal_has_been_found and episode_reward > 0 and episode >= 50:
            print(f"\nZIEL IN EPISODE {episode + 1} GEFUNDEN.\n")
            self.goal_has_been_found = True
        
        # Führe das Rückwärts-Lernen durch
        self._learn_backwards()
        
        self.epsilon = self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)

    def _learn_backwards(self):
        all_known_states = set(self.model.keys())
        
        for _ in range(len(all_known_states)): # Iteriere oft genug für Konvergenz
            for state in self.model:
                for action in self.model[state]:
                    
                    # --- PROBABILISTISCHES Q-VALUE UPDATE ---
                    outcomes = self.model[state][action]
                    total_transitions = sum(outcomes.values())
                    if total_transitions == 0:
                        continue

                    expected_q_value = 0
                    for next_state, count in outcomes.items():
                        # Berechne Wahrscheinlichkeit P(s' | s, a)
                        probability = count / total_transitions
                        
                        # Berechne durchschnittliche Belohnung R(s, a, s')
                        avg_reward = np.mean(self.rewards[state][action][next_state])
                        
                        # Berechne den Wert dieses Zweigs
                        next_max = np.max(self._get_q_values(next_state))
                        branch_value = probability * (avg_reward + self.discount_rate * next_max)
                        
                        expected_q_value += branch_value
                    
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(self.n_actions)
                    self.q_table[state][action] = expected_q_value