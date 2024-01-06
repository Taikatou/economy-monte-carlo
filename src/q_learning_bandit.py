import numpy as np
import random
import matplotlib.pyplot as plt

class ContextualBandit:
    def __init__(self):
        self.resources = {1: 'Wood', 2: 'Metal', 3: 'Gem', 4: 'DragonScale'}
        self.arm_probabilities = {
            0: {1: 0.5, 2: 0.2, 3: 0.2, 4: 0.1},
            1: {1: 0.1, 2: 0.6, 3: 0.2, 4: 0.1},
            2: {1: 0.2, 2: 0.1, 3: 0.6, 4: 0.1},
            3: {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.7}
        }
        # Initialize Q-table
        self.num_arms = len(self.arm_probabilities)
        self.num_contexts = len(self.resources)
        self.q_table = np.zeros((self.num_contexts, self.num_arms))
        self.recent_regrets = []
        self.avg_regrets = []
        self.recent_rewards = []
        self.avg_rewards = []

    def calculate_regret(self, context):
        # Calculate the regret for a given context
        best_arm = self.best_possible_arm(context)
        best_reward = self.arm_probabilities[best_arm][context]
        return best_reward

    def update_regret(self, chosen_arm, context):
        # Calculate and update recent regrets
        best_reward = self.calculate_regret(context)
        actual_reward = self.arm_probabilities[chosen_arm][context]
        regret = best_reward - actual_reward

        # Update the recent regrets list
        self.recent_regrets.append(regret)
        if len(self.recent_regrets) >= 30:
            # If there are at least 30 regrets, calculate the average
            self.avg_regrets.append(np.mean(self.recent_regrets[-30:]))
    
    def context_sampling(self):
        context = np.random.randint(1, 5)
        return context

    def choose_arm(self, context, epsilon=0.1):
        # Implementing epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            return np.argmax(self.q_table[context - 1])
    
    def best_possible_arm(self, context):
        # Determine the arm with the highest probability of reward for the given context
        return max(self.arm_probabilities, key=lambda arm: self.arm_probabilities[arm][context])
      
    def expected_reward_oracle(self, context):
        # Calculate the expected reward for the best possible arm
        best_arm = self.best_possible_arm(context)
        return self.arm_probabilities[best_arm][context]
    
    def get_reward(self, arm, context):
        # Calculate cumulative probabilities for each resource
        cumulative_probs = np.cumsum([self.arm_probabilities[arm][res] for res in range(1, 5)])
        
        # Generate a random number
        rand_val = np.random.rand()

        # Determine which resource is selected based on the cumulative probabilities
        for i, cum_prob in enumerate(cumulative_probs):
            if rand_val < cum_prob:
                # Reward is 1 if the selected resource matches the context, else 0
                return int(i + 1 == context)
        return 0
    
    def update_q_table(self, context, arm, reward, learning_rate=0.1):
        current_q_value = self.q_table[context - 1, arm]
        new_q_value = current_q_value + learning_rate * (reward - current_q_value)
        self.q_table[context - 1, arm] = new_q_value
    
    def update_total_reward(self, reward):
        # Update the recent rewards list
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) >= 30:
            # If there are at least 30 rewards, calculate the average
            self.avg_rewards.append(np.mean(self.recent_rewards[-30:]))

# Example usage
bandit = ContextualBandit()
epochs = 1000

for _ in range(epochs):
    context = bandit.context_sampling()
    arm = bandit.choose_arm(context)
    reward = bandit.get_reward(arm, context)
    bandit.update_q_table(context, arm, reward)
    bandit.update_regret(arm, context)
    bandit.update_total_reward(reward)

# Plot the average regrets
plt.plot(bandit.avg_regrets)
plt.title("Average Regret over Last 10 Decisions")
plt.xlabel("Epochs")
plt.ylabel("Average Regret")
plt.show()

plt.plot(bandit.avg_rewards)
plt.title("Total Reward over Time")
plt.xlabel("Epochs")
plt.ylabel("Cumulative Reward")
plt.show()