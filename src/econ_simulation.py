import resource_utils
import market_utils


def run_simulation_for_selected_swords(selected_swords, trained_bandit, prices, budget, max_iterations=20000, num_runs=3):
    total_results = []

    for _ in range(num_runs):
        grid_search_results = {sword: {} for sword in selected_swords}

        for sword, requirements in selected_swords.items():
            # Convert requirements to context format for the bandit
            context = [0 if resource not in requirements else requirements[resource] for resource in ["wood", "metal", "gem", "dragonscale"]]

            # Run the market simulation with the current context until all resources are collected
            while not all(req == 0 for req in context):
                market_data = market_utils.generate_variable_market_resources(trained_bandit, prices, budget, max_iterations, context)
                # Update context based on acquired resources
                for resource, count in requirements.items():
                    if market_data[resource].iloc[-1] >= count:
                        context[["wood", "metal", "gem", "dragonscale"].index(resource)] = 0

            # Record results for this set of requirements
            grid_search_results[sword] = market_data

        total_results.append(grid_search_results)

    # Return the average results across runs
    # Note: Further processing might be needed to extract meaningful average results
    avg_results = {}  # Define a method to calculate average results from total_results
    return avg_results