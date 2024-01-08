from random import random
import numpy as np


def generate_variable_market_resources(probabilities, prices, budget, max_iterations):
    resources = [resource for resource in probabilities.keys() if resource != 'none']
    bought_counts = {resource: [0] for resource in resources}
    current_budget = budget

    for _ in range(max_iterations):
        adjusted_probabilities = {res: prob * random.uniform(0.8, 1.2) for res, prob in probabilities.items()}
        total_prob = sum(adjusted_probabilities.values())
        adjusted_probabilities = {res: prob / total_prob for res, prob in adjusted_probabilities.items()}
        adjusted_prices = {res: price * random.uniform(0.8, 1.2) for res, price in prices.items()}

        market_resource = np.random.choice(resources + ['none'], p=list(adjusted_probabilities.values()))
        if market_resource != 'none':
            resource_price = adjusted_prices[market_resource]

            if current_budget >= resource_price:
                current_budget -= resource_price
                bought_counts[market_resource][-1] += 1

        for resource in resources:
            bought_counts[resource].append(bought_counts[resource][-1])

    return pd.DataFrame(bought_counts)