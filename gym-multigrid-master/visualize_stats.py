import argparse
import matplotlib.pyplot as plt
import matplotlib
from config import ENV_CLASS, ENV_RULE_SETS
from plot import exponential_smoothing
import os
from os.path import join, dirname
import pickle

import numpy as np 
from collections import Counter

matplotlib.use("Agg")


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="simple",
                    help="name of the environment (REQUIRED): simple, lava, lava2, key")
parser.add_argument("--model_dir",  default=join(dirname(os.path.abspath(__file__)), "results_eval_viz"), type=str,
                    help="model_dir")

args = parser.parse_args()


episode_rewards = {}
num_agents_died = {}
wall_hits = {}
num_balls_collected = {}
PATH = join(args.model_dir, args.env)

for file in os.listdir(PATH):
    settings = file.split("_")
    model_type = settings[1]
    USE_KG = "USEKG" in settings
    if USE_KG:
        ruleset= settings[settings.index("USEKG") +1]
    else:
        ruleset = ""
    
    use_rules_str= f", rulesest: {ruleset}" if USE_KG else ""
    label = f"{model_type}" + use_rules_str

    with open(join(PATH,file), "rb") as handle:
        data_plot = pickle.load(handle)
    
    episode_rewards[label] = data_plot["Total Reward"]
    num_agents_died[label] = data_plot["Num Agents died"]
    wall_hits[label] = data_plot["Num Wall Hits"] 
    num_balls_collected[label] = data_plot["Num Balls collected"]  

#  Plot 

colors = plt.get_cmap("tab20").colors  

# fig, ax = plt.subplots(figsize=(10, 6))
# for c,(label, data) in enumerate(episode_rewards.items()):
#     # Create plot

#     # Compute and plot smoothed values
#     ma = exponential_smoothing(data, alpha=0.3)
#     ma_idx_start = len(data) - len(ma)
#     ax.plot(range(ma_idx_start, len(data)), ma, label=label,
#             linestyle="-", color=colors[c], linewidth=2)

#     # # Shade the area under the original data
#     # ax.fill_between(range(len(data)), data, color=colors[c], alpha=0.1)

# # Add labels and legend
# ax.set_xlabel("Episode")
# ax.set_ylabel("Rewards per episodes")
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
# plt.savefig(f"{args.env}_episode_rewards_plot.png", bbox_inches='tight') # Save the
# plt.close()

# Define the desired legend order
desired_legend_order = ['COMA', 'COMA, rulesest: ball only', 'COMA, rulesest: ball with search strats', 'COMA, rulesest: conflicting rules', 'COMA, rulesest: irrelevant rules', 
                        'COMA, rulesest: lava + ball',
                        'COMA, rulesest: ball + key',
                        'MAPPO', 'MAPPO, rulesest: ball only', 'MAPPO, rulesest: ball with search strats', 'MAPPO, rulesest: conflicting rules', 'MAPPO, rulesest: irrelevant rules',  
                        'MAPPO, rulesest: lava + ball', 
                        'MAPPO, rulesest: ball + key',
                        'random',
                        'expert, rulesest: ball only', 'expert, rulesest: ball with search strats', 'expert, rulesest: conflicting rules', 'expert, rulesest: irrelevant rules',
                        'expert, rulesest: lava + ball',
                        'expert, rulesest: ball + key']  


fig, ax = plt.subplots(figsize=(10, 6))
lines = []  # To store line objects
labels = []  # To store corresponding labels

for c, (label, data) in enumerate(episode_rewards.items()):
    # Compute and plot smoothed values
    ma = exponential_smoothing(data, alpha=0.3)
    ma_idx_start = len(data) - len(ma)
    line, = ax.plot(range(ma_idx_start, len(data)), ma, label=label,
                    linestyle="-", color=colors[c], linewidth=2)
    
    # Store line and label only if it's in the desired legend order
    if label in desired_legend_order:
        lines.append(line)
        labels.append(label)

# Reorder legend dynamically based on existing labels
ordered_handles = [lines[labels.index(label)] for label in desired_legend_order if label in labels]
ordered_labels = [label for label in desired_legend_order if label in labels]

ax.set_xlabel("Episode")
ax.set_ylabel("Rewards per episodes")

# Apply reordered legend
ax.legend(ordered_handles, ordered_labels, loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot
plt.savefig(f"{args.env}_episode_rewards_plot.png", bbox_inches='tight')
plt.close()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Categories and positions
categories = ['0 died', '1 died', '2 died', '3 died', '4 died']
y_positions = np.arange(len(categories))

# Bar height calculation
bar_height = 0.8 / len(num_agents_died)

# Track handles and labels for legend
handles = []
labels = []

# Plotting each rule set
for c, (label, num_agents_died_arr) in enumerate(num_agents_died.items()):
    # Count occurrences of each category (0, 1, 2, 3, 4 died)
    counts = np.array([Counter(num_agents_died_arr).get(i, 0) for i in range(len(categories))])
    
    # Offset bars for each rule set
    bar = ax.barh(y_positions - (len(num_agents_died) / 2 - c) * bar_height, counts,
            height=bar_height, label=label, color=colors[c % len(colors)])
    
    if label in desired_legend_order:
        handles.append(bar)
   
        labels.append(label)

ordered_handles = [handles[labels.index(label)] for label in desired_legend_order if label in labels]
ordered_labels = [label for label in desired_legend_order if label in labels]

# Labels and title
ax.set_xlabel('Num Episodes')
ax.set_ylabel('Num Agents Died')
ax.set_title('Num Agents Died per Episode')
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)


ax.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show or save the plot
plt.tight_layout()
plt.savefig(f"{args.env}_num_agents_died_plot.png")
plt.close()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Track handles and labels for legend
handles = []
labels = []

# Categories and positions
categories = ['0 collected', '1 ball collected', '2 balls collected', '3 balls collected', '5 balls collected']
y_positions = np.arange(len(categories))

# Bar height calculation
bar_height = 0.8 / len(num_balls_collected)

# Plotting each rule set
for c, (label, num_balls_collected_arr) in enumerate(num_balls_collected.items()):
    # Count occurrences of each category (0, 1, 2, 3, 4, 5 collected)
    counts = np.array([Counter(num_balls_collected_arr).get(i, 0) for i in range(len(categories))])
    
    # Offset bars for each rule set
    bar = ax.barh(y_positions - (len(num_balls_collected) / 2 - c) * bar_height, counts,
            height=bar_height, label=label, color=colors[c % len(colors)])

    if label in desired_legend_order:
        handles.append(bar)
        labels.append(label)

ordered_handles = [handles[labels.index(label)] for label in desired_legend_order if label in labels]
ordered_labels = [label for label in desired_legend_order if label in labels]

# Labels and title
ax.set_xlabel('Num Episodes')
ax.set_ylabel('Num Balls Collected')
ax.set_title('Num Balls Collected per Episode')
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)
ax.legend(ordered_handles, ordered_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

# Show or save the plot
plt.tight_layout()
plt.savefig(f"{args.env}_num_balls_collected_plot.png")
plt.close()

from matplotlib.ticker import MaxNLocator

# # Example data
# wall_hits = {
#     "e1": [121, 123, 1, 23],
#     "e2": [812, 23, 57, 490],
#     "e3": [83, 2383, 83],
#     "e4": [32, 120, 47, 305],
#     "e5": [56, 34, 19, 90],
#     "e6": [290, 580, 100]
# }


# Calculate the maximum number of episodes (i.e., the max length of any wall_hits list)
max_episodes = max(len(hits) for hits in wall_hits.values())

# Plot overlapping histograms
plt.figure(figsize=(10, 6))

# Track handles and labels for the legend
handles = []
labels = []

for i, (label, data) in enumerate(wall_hits.items()):
    hist = plt.hist(data, bins=10, alpha=0.5, label=label, color=colors[i % len(colors)])

    if label in desired_legend_order:
        handles.append((hist[2][0], label))

# Reorder legend dynamically based on desired_legend_order
ordered_handles_labels = [(h, l) for l in desired_legend_order for h, lbl in handles if lbl == l]
ordered_handles, ordered_labels = zip(*ordered_handles_labels) if ordered_handles_labels else ([], [])

# Set y-axis to use only integer ticks and reflect max episodes
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(0, max_episodes//2)  # Set the y-axis limit to max episodes

# Add labels and legend
plt.xlabel("Wall Hits")
plt.ylabel("Num of Episodes")
plt.title("Wall Hits Episodes")
plt.legend(ordered_handles, ordered_labels)

# Display the plot
plt.savefig(f"{args.env}_wall_hits_plot.png")
