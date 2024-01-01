import resource_utils
import market_utils

def run_simulation_for_selected_swords(selected_swords, probabilities, prices, budget, max_iterations=20000, num_runs=3):
    total_results = []

    for _ in range(num_runs):
        market_data = generate_variable_market_resources(probabilities, prices, budget, max_iterations)
        grid_search_results = {sword: {} for sword in selected_swords}

        for sword, requirements in selected_swords.items():
            for resource in ["wood", "metal", "gem", "dragonscale"]:
                if resource in requirements:
                    adjusted_requirements = requirements.copy()
                    adjusted_requirements[resource] -= 1
                    reduced_crafted = simulate_crafting_with_market({sword: adjusted_requirements}, market_data)
                    grid_search_results[sword][f"Reduced {resource}"] = reduced_crafted[sword]

                    adjusted_requirements = requirements.copy()
                    adjusted_requirements[resource] += 1
                    increased_crafted = simulate_crafting_with_market({sword: adjusted_requirements}, market_data)
                    grid_search_results[sword][f"Increased {resource}"] = increased_crafted[sword]

        total_results.append(pd.DataFrame.from_dict(grid_search_results, orient='index'))

    avg_results = pd.concat(total_results).groupby(level=0).mean()
    return avg_results