import matplotlib.pyplot as plt

# Extracting only the "Original Steps" columns from both tables for comparison
original_steps = combined_simulation_results["Original Steps"]
reduced_steps = reduced_simulation_results["Original Steps"]

# Normalizing the values
original_steps_normalized = original_steps / original_steps.max()
reduced_steps_normalized = reduced_steps / reduced_steps.max()

# Plotting the comparative difficulty change
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(original_steps_normalized))

bar1 = ax.bar(index, original_steps_normalized, bar_width, label='Original')
bar2 = ax.bar(index + bar_width, reduced_steps_normalized, bar_width, label='Reduced')

ax.set_xlabel('Sword Type')
ax.set_ylabel('Normalized Difficulty')
ax.set_title('Comparative Difficulty Change for Sword Crafting')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(original_steps_normalized.index)
ax.legend()