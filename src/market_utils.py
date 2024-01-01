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