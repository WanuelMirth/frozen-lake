# agents/rmax_agent.py
import numpy as np

class RMaxAgent:
    def __init__(self, observation_space, action_space, params):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_actions = action_space.n
        self.num_states = observation_space.n
        
        self.discount_rate = params["discount_rate"]
        self.m = int(params.get("m", 5))  # Confidence/known threshold
        self.r_max = float(params.get("r_max", 1.0))
        
        # Calculate V_max = R_max / (1 - gamma)
        self.v_max = self.r_max / (1.0 - self.discount_rate)
        
        # Initialize Q-table with V_max
        self.q_table = np.full((self.num_states, self.n_actions), self.v_max, dtype=float)
        
        # Model counters
        # n_sa[(s, a)] = visit count of state-action pair
        self.n_sa = {}
        # n_sas[(s, a, s')] = transition count
        self.n_sas = {}
        # r_sas[(s, a, s')] = sum of rewards
        self.r_sas = {}
        
        # Keep track of observed transitions for fast lookup
        self.outcomes = {}
        
        # Terminal states
        self.terminal_states = set()
        
        # Visited states (source states)
        self.visited_states = set()
        
        print(f"R-Max Agent initialized: m={self.m}, discount={self.discount_rate:.4f}, v_max={self.v_max:.4f}")

    def choose_action(self, state):
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        # Find all actions that achieve max Q-value within small tolerance
        best_actions = np.where(np.isclose(q_values, max_q, atol=1e-8))[0]
        return np.random.choice(best_actions)

    def learn(self, state, action, reward, new_state, terminated):
        pair = (state, action)
        self.n_sa[pair] = self.n_sa.get(pair, 0) + 1
        
        transition = (state, action, new_state)
        self.n_sas[transition] = self.n_sas.get(transition, 0) + 1
        self.r_sas[transition] = self.r_sas.get(transition, 0.0) + float(reward)
        
        self.visited_states.add(state)
        
        if pair not in self.outcomes:
            self.outcomes[pair] = set()
        self.outcomes[pair].add(new_state)
        
        if terminated:
            self.terminal_states.add(new_state)
            # Ensure Q-values of terminal states are strictly 0.0
            self.q_table[new_state] = 0.0

    def on_episode_end(self, episode, episode_reward):
        self._learn_backwards()

    def _learn_backwards(self):
        theta = 1e-4
        if not self.visited_states:
            return
            
        max_iterations = len(self.visited_states) * 10
        
        for _ in range(max_iterations):
            max_delta = 0
            for state in self.visited_states:
                if state in self.terminal_states:
                    continue
                
                for action in range(self.n_actions):
                    pair = (state, action)
                    n_sa = self.n_sa.get(pair, 0)
                    
                    if n_sa < self.m:
                        new_q = self.v_max
                    else:
                        expected_value = 0.0
                        for next_state in self.outcomes.get(pair, set()):
                            transition = (state, action, next_state)
                            count = self.n_sas.get(transition, 0)
                            prob = count / n_sa
                            avg_reward = self.r_sas.get(transition, 0.0) / count
                            
                            if next_state in self.terminal_states:
                                next_val = 0.0
                            else:
                                next_val = np.max(self.q_table[next_state])
                                
                            expected_value += prob * (avg_reward + self.discount_rate * next_val)
                        new_q = expected_value
                    
                    old_q = self.q_table[state][action]
                    self.q_table[state][action] = new_q
                    max_delta = max(max_delta, abs(old_q - new_q))
            
            if max_delta < theta:
                break
