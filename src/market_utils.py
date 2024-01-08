import numpy as np
import pandas as pd
import random

def generate_market_resources(probabilities, prices, budget, max_iterations):
    resources = list(probabilities.keys())
    bought_counts = {resource: [0] for resource in resources}
    current_budget = budget

    for _ in range(max_iterations):
        generated_resource = np.random.choice(resources, p=list(probabilities.values()))
        resource_price = prices[generated_resource]

        if current_budget >= resource_price:
            current_budget -= resource_price
            bought_counts[generated_resource][-1] += 1

        for resource in resources:
            bought_counts[resource].append(bought_counts[resource][-1])

    return pd.DataFrame(bought_counts)

def simulate_crafting_with_market(sword_requirements, market_data):
    results = {}

    for sword, requirements in sword_requirements.items():
        crafted = market_data.apply(lambda x: all(x[resource] >= count for resource, count in requirements.items()), axis=1)
        try:
            results[sword] = crafted.idxmax() + 1
        except ValueError:
            results[sword] = np.nan

    return results

def generate_variable_market_resources(trained_bandit, prices, budget, max_iterations, context):
    resources = ["wood", "metal", "gem", "dragonscale", "none"]
    bought_counts = {resource: [0] for resource in resources}
    current_budget = budget

    for _ in range(max_iterations):
        # Use the trained bandit to choose an area based on the current context
        chosen_arm = trained_bandit.choose_arm(context)

        # Generate resources based on the chosen arm's probabilities
        arm_probabilities = trained_bandit.arm_probabilities[chosen_arm]
        generated_resource = np.random.choice(resources, p=list(arm_probabilities.values()))

        # Buy the resource if there's enough budget
        if generated_resource != 'none' and current_budget >= prices[generated_resource]:
            current_budget -= prices[generated_resource]
            bought_counts[generated_resource][-1] += 1

        # Record the current state
        for resource in resources:
            bought_counts[resource].append(bought_counts[resource][-1])

    return pd.DataFrame(bought_counts)