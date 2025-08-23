# agents/rbql_agent.py
import numpy as np
from collections import deque

class RBQLAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.discount_rate = params["discount_rate"]
        
        self.q_table = {} 
        self.model = {}
        
        # NEU: Epsilon-Parameter für die Hybrid-Exploration hinzufügen
        self.epsilon = params["max_epsilon"]
        self.min_epsilon = params["min_epsilon"]
        self.max_epsilon = params["max_epsilon"]
        self.epsilon_decay_rate = params["epsilon_decay_rate"]
        
        self.episode_trajectory = []
        print("Recursive Backwards Q-Learning (RBQL) Agent initialisiert (mit Hybrid-Exploration).")

    def _get_q_values(self, state):
        return self.q_table.get(state, np.zeros(self.n_actions))

    def choose_action(self, state):
        # --- NEUE HYBRID-LOGIK ---
        
        # 1. Priorität: Gezielte Exploration (wie im Paper beschrieben)
        known_actions_for_state = self.model.get(state, {})
        unexplored_actions = [a for a in range(self.n_actions) if a not in known_actions_for_state]
        
        if unexplored_actions:
            # Wenn es eine unerforschte Tür gibt, gehe hindurch
            return np.random.choice(unexplored_actions)
        
        # 2. Priorität: Epsilon-Greedy, wenn alle Wege vom aktuellen Ort bekannt sind
        # Dies verhindert das Steckenbleiben in bekannten Schleifen.
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Zufällige Aktion zum Ausbrechen
        else:
            return np.argmax(self._get_q_values(state))  # Beste bekannte Aktion

    def learn(self, state, action, reward, new_state):
        if state not in self.model:
            self.model[state] = {}
        if action not in self.model[state]:
            self.model[state][action] = (reward, new_state)
        
        self.episode_trajectory.append((state, action))

    def on_episode_end(self, episode):
        # NEU: Epsilon-Decay auch für RBQL
        self.epsilon = self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)
            
        # Führe das Rückwärts-Lernen durch
        if not self.episode_trajectory:
            return
        
        all_known_states = set(self.model.keys())
        
        # Iterative Updates
        for _ in range(len(all_known_states) * 2):
            for state in self.model:
                for action in self.model[state]:
                    reward, next_state = self.model[state][action]
                    next_max = np.max(self._get_q_values(next_state))
                    
                    new_value = reward + self.discount_rate * next_max
                    
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(self.n_actions)
                    self.q_table[state][action] = new_value

        self.episode_trajectory = []