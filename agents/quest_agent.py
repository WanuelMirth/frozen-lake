# agents/quest_agent.py
import numpy as np

class QUESTAgent:
    def __init__(self, observation_space, action_space, params):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.discount_rate = params["discount_rate"]
        
        self.q_table = {} 
        self.model = {}
        self.rewards = {}
        
        self.n_s = {}  # Counts visits to state s: N(s)
        self.n_sa = {} # Counts execution of action a in state s: N(s,a)
        # c is a hyperparameter controlling agent curiosity
        self.exploration_constant_c = params.get("exploration_constant_c", 1.4) 
        
        print(f"QUEST Agent initialized with c={self.exploration_constant_c}.")

    def _get_q_values(self, state):
        return self.q_table.get(state, np.zeros(self.n_actions))

    def choose_action(self, state):
        # If a state is completely new, initialize counters and explore randomly
        if state not in self.n_s:
            self.n_s[state] = 0
            self.n_sa[state] = np.zeros(self.n_actions)
            return np.random.randint(self.n_actions)

        uct_values = np.zeros(self.n_actions)
        q_values = self._get_q_values(state)

        for action in range(self.n_actions):
            # If an action has never been executed, select it immediately.
            # This is crucial to ensure all actions are explored.
            if self.n_sa[state][action] == 0:
                return action
            
            # 1. Exploitation term: Use knowledge from the Q-table (learned)
            exploitation_term = q_values[action]
            
            # 2. Exploration term: Be curious about less frequently chosen actions
            exploration_term = self.exploration_constant_c * \
                np.sqrt(np.log(self.n_s[state]) / self.n_sa[state][action])
            
            uct_values[action] = exploitation_term + exploration_term
            
        # Select the action with the highest UCB value
        return np.argmax(uct_values)

    def learn(self, state, action, reward, new_state):
        # --- MODEL UPDATE ---
        if state not in self.model:
            self.model[state] = {a: {} for a in range(self.n_actions)}
            self.rewards[state] = {a: {} for a in range(self.n_actions)}
        
        outcomes = self.model[state][action]
        outcomes[new_state] = outcomes.get(new_state, 0) + 1
        
        # Optimize reward tracking: store directly as float scalar (no list append)
        self.rewards[state][action][new_state] = float(reward)

        # --- VISIT COUNTER UPDATE FOR UCB ---
        self.n_s[state] = self.n_s.get(state, 0) + 1
        if state not in self.n_sa:
             self.n_sa[state] = np.zeros(self.n_actions)
        self.n_sa[state][action] += 1

    def on_episode_end(self, episode, episode_reward):
        self._learn_backwards()

    def _learn_backwards(self):
        all_known_states = list(self.model.keys())
        if not all_known_states:
            return
            
        # Iterate enough times to guarantee convergence, with early stopping
        theta = 1e-4
        max_iterations = len(all_known_states) * 5
        
        for _ in range(max_iterations):
            max_delta = 0
            for state in all_known_states:
                for action in range(self.n_actions):
                    
                    if action not in self.model[state] or not self.model[state][action]:
                        continue # This action has never been executed from this state
                    
                    outcomes = self.model[state][action]
                    total_transitions = sum(outcomes.values())
                    if total_transitions == 0:
                        continue

                    expected_q_value = 0
                    for next_state, count in outcomes.items():
                        probability = count / total_transitions
                        # Direct lookup (avoids slow np.mean call)
                        avg_reward = self.rewards[state][action][next_state]
                        next_max = np.max(self._get_q_values(next_state))
                        branch_value = probability * (avg_reward + self.discount_rate * next_max)
                        expected_q_value += branch_value
                    
                    if state not in self.q_table:
                        self.q_table[state] = np.zeros(self.n_actions)
                        
                    old_val = self.q_table[state][action]
                    self.q_table[state][action] = expected_q_value
                    max_delta = max(max_delta, abs(old_val - expected_q_value))
                    
            if max_delta < theta:
                break
