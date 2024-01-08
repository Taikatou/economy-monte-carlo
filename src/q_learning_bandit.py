import numpy as np
import random
import matplotlib.pyplot as plt

from econ_simulation import run_simulation_for_selected_swords

from enum import IntEnum


# class syntax
class Resource(IntEnum):
    NOTHING = 0
    WOOD = 1
    METAL = 2
    GEM = 3
    DRAGONSCALE = 4


class ContextualBandit:
    def __init__(self, arm_probabilities):
        self.resources = {Resource.WOOD: 'Wood', Resource.METAL: 'Metal', Resource.GEM: 'Gem', Resource.DRAGONSCALE: 'DragonScale'}
        self.arm_probabilities = arm_probabilities 
        # Initialize Q-table
        self.num_arms = len(self.arm_probabilities)
        self.num_contexts = len(self.resources)
        self.q_table = np.zeros((625, self.num_arms))
        self.recent_regrets = []
        self.avg_regrets = []
        self.recent_rewards = []
        self.avg_rewards = []
        self.new_context = []

    def calculate_regret(self, context):
        # Calculate the regret for a given context
        best_arm = self.best_possible_arm(context)
        best_reward = self.arm_probabilities[best_arm][context]
        return best_reward

    def context_to_state(self, context):
        # Corrected state calculation as a base-5 number
        state = sum(context[i] * (5 ** i) for i in range(4))
        return state

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
        context = np.random.randint(0, 5, size=4)
        return context

    def choose_arm(self, context, epsilon=0.1):
        # Implementing epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(0, self.num_arms)
        else:
            state = self.context_to_state(context)
            return np.argmax(self.q_table[state])
    
    def best_possible_arm(self, context):
        # Determine the arm with the highest probability of reward for the given context
        return max(self.arm_probabilities, key=lambda arm: self.arm_probabilities[arm][context])
      
    def expected_reward_oracle(self, context):
        # Calculate the expected reward for the best possible arm
        best_arm = self.best_possible_arm(context)
        return self.arm_probabilities[best_arm][context]
    
    def get_reward(self, arm, context):
        resource = int(np.random.choice([Resource.NOTHING, Resource.WOOD, Resource.METAL, Resource.GEM, Resource.DRAGONSCALE], p=[self.arm_probabilities[arm][i] for i in range(0, 5)]))
        if resource == 0:
            return 0
        self.new_context = context
        if self.new_context[resource - 1] > 0:
            self.new_context[resource - 1] -= 1
        return int(np.all(self.new_context == 0))
    
    def update_q_table(self, context, arm, reward, learning_rate=0.1):
        state = self.context_to_state(context)
        current_q_value = self.q_table[state, arm]
        new_q_value = current_q_value + learning_rate * (reward - current_q_value)
        self.q_table[state, arm] = new_q_value

config = {  # Configuration 1
    0: {0: 0.25, 1: 0.5, 2: 0.25, 3: 0.0, 4: 0.0},  # Forest
    1: {0: 0.25, 1: 0.30, 2: 0.30, 3: 0.15, 4: 0.0},  # Mountain
    2: {0: 0.6, 1: 0.10, 2: 0.10, 3: 0.20, 4: 0.0},  # Sea
    3: {0: 0.45, 1: 0.15, 2: 0.15, 3: 0.15, 4: 0.10}  # Volcano
}

# Example usage
bandit = ContextualBandit(config)
epochs = 1000

for e in range(epochs):
    print(e)
    context = bandit.context_sampling()
    contexts = []
    reward = 0
    discount = 0.999
    while sum(context) != 0:
        arm = bandit.choose_arm(context)
        new_reward = bandit.get_reward(arm, context)
        reward += new_reward * discount
        discount = discount * 0.995
        contexts.append(context)
        context = bandit.new_context

    print(reward)
    for c in contexts:
        bandit.update_q_table(c, arm, reward)
    # bandit.update_regret(arm, context)
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

adjusted_prices = {"wood": 1, "metal": 2, "gem": 5, "dragonscale": 10}
adjusted_budget = 1000  # Increased budget
max_iterations = 1000

sword_requirements = {
    "Beginner Sword": {"wood": 1, "metal": 1},
    "Intermediate Sword": {"wood": 2, "metal": 2},
    "Advanced Sword": {"wood": 1, "metal": 1, "gem": 1},
    "Epic Sword": {"wood": 2, "metal": 2, "gem": 2},
    "Ultimate Sword": {"wood": 2, "metal": 2, "gem": 2, "dragonscale": 1}
}

results = run_simulation_for_selected_swords(sword_requirements, bandit, adjusted_prices, adjusted_budget, max_iterations)
print(results)

# Print or analyze the results
# Note: The print statement is for demonstration purposes; actual analysis may vary
print(results)