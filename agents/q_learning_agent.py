# agents/q_learning_agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.q_table = np.zeros((observation_space.n, action_space.n))
        
        self.learning_rate = params["learning_rate"]
        self.discount_rate = params["discount_rate"]
        self.epsilon = params["max_epsilon"]
        self.min_epsilon = params["min_epsilon"]
        self.max_epsilon = params["max_epsilon"]
        self.epsilon_decay_rate = params["epsilon_decay_rate"]
        
        print("Standard Q-Learning Agent initialisiert.")

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()  # Exploration: Zufällige Aktion
        else:
            return np.argmax(self.q_table[state, :])  # Exploitation: Beste bekannte Aktion

    def learn(self, state, action, reward, new_state):
        """Klassische Q-Learning Update-Formel wird bei jedem Schritt ausgeführt."""
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[new_state, :])
        
        new_value = old_value + self.learning_rate * (reward + self.discount_rate * next_max - old_value)
        self.q_table[state, action] = new_value

    def on_episode_end(self, episode, episode_reward):
        """Wird am Ende jeder Episode aufgerufen, um Epsilon zu verringern."""
        self.epsilon = self.min_epsilon + \
            (self.max_epsilon - self.min_epsilon) * np.exp(-self.epsilon_decay_rate * episode)