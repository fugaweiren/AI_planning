import argparse
import matplotlib.pyplot as plt
import matplotlib

import os
from os.path import join, dirname
import pickle

import numpy as np 
from collections import Counter
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="simple",
                    help="name of the environment (REQUIRED): simple, lava, lava2, key")
parser.add_argument("--model_dir",  default=join(dirname(os.path.abspath(__file__)), "results_eval_viz"), type=str,
                    help="model_dir")

args = parser.parse_args()
args.env = "key"


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
    
    use_rules_str= f", {ruleset}" if USE_KG else ""
    label = f"{model_type}" + use_rules_str

    with open(join(PATH,file), "rb") as handle:
        data_plot = pickle.load(handle)
    
    episode_rewards[label] = data_plot["Total Reward"]
    num_agents_died[label] = data_plot["Num Agents died"]
    wall_hits[label] = data_plot["Num Wall Hits"] 
    num_balls_collected[label] = data_plot["Num Balls collected"]  

#  Plot 

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

sublabel = ["COMA", "MAPPO", "others"]
for s in sublabel:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    if s == "COMA" or s == "MAPPO":
        data_dict ={ x:y for x,y in episode_rewards.items() if s in x}
    else:
        
        data_dict ={ x:y for x,y in episode_rewards.items() if "expert" in x or "random" in x}
    
    color_map= {label: colors[i] for i, label in enumerate(data_dict)}
    max_count = 0
    for c,(label, data) in enumerate(data_dict.items()):
        # Plot overlapping histograms
        counts, _, _ =ax.hist(data, bins=50, alpha=0.5, label=label, color=color_map[label])
        max_count = max(max_count, max(counts))


    ax.set_ylim(0, max_count)  # Set the y-axis limit to max episodes
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Add labels and legend

    ax.set_xlabel("Rewards per episode")
    ax.set_ylabel("Episodes")
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f"{args.env}_{s}_episode_rewards_plot.png", bbox_inches='tight') # Save the
    plt.close()




colors = plt.get_cmap("tab20", 15).colors
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Categories and positions
categories = ['0 died', '1 died', '2 died', '3 died', '4 died']
y_positions = np.arange(len(categories))

# Bar height calculation
bar_height = 0.8 / len(num_agents_died)

# Plotting each rule set
for c, (label, num_agents_died_arr) in enumerate(num_agents_died.items()):
    # Count occurrences of each category (0, 1, 2, 3, 4 died)
    counts = np.array([Counter(num_agents_died_arr).get(i, 0) for i in range(len(categories))])
    
    # Offset bars for each rule set
    ax.barh(y_positions - (len(num_agents_died) / 2 - c) * bar_height, counts,
            height=bar_height, label=label, color=colors[c % len(colors)])

# Labels and title
ax.set_xlabel('Num Episodes')
ax.set_ylabel('Num Agents Died')
ax.set_title('Num Agents Died per Episode')
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)

handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
sorted_labels, sorted_handles = zip(*sorted_handles_labels)
ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

# Show or save the plot
plt.tight_layout()
plt.savefig(f"{args.env}_num_agents_died_plot.png")
plt.close()



colors = plt.get_cmap("tab20", 15).colors
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Categories and positions
categories = ['0 collected', '1 ball collected', '2 balls collected', '3 balls collected', '4 balls collected' ,'5 balls collected']
y_positions = np.arange(len(categories))

# Bar height calculation
bar_height = 0.8 / len(num_balls_collected)

# Plotting each rule set
for c, (label, num_balls_collected_arr) in enumerate(num_balls_collected.items()):
    # Count occurrences of each category (0, 1, 2, 3, 4, 5 collected)
    counts = np.array([Counter(num_balls_collected_arr).get(i, 0) for i in range(len(categories))])
    
    # Offset bars for each rule set
    ax.barh(y_positions - (len(num_balls_collected) / 2 - c) * bar_height, counts,
            height=bar_height, label=label, color=colors[c % len(colors)])

# Labels and title
ax.set_xlabel('Num Episodes')
ax.set_ylabel('Num Balls Collected')
ax.set_title('Num Balls Collected per Episode')
ax.set_yticks(y_positions)
ax.set_yticklabels(categories)
handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
sorted_labels, sorted_handles = zip(*sorted_handles_labels)
ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1.05, 1))

# Show or save the plot
plt.tight_layout()
plt.savefig(f"{args.env}_num_balls_collected_plot.png")
plt.close()


# # Example data
# wall_hits = {
#     "e1": [121, 123, 1, 23],
#     "e2": [812, 23, 57, 490],
#     "e3": [83, 2383, 83],
#     "e4": [32, 120, 47, 305],
#     "e5": [56, 34, 19, 90],
#     "e6": [290, 580, 100]
# }


colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for s in sublabel:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if s == "COMA" or s == "MAPPO":
        data_dict ={ x:y for x,y in wall_hits.items() if s in x}
    else:
        
        data_dict ={ x:y for x,y in wall_hits.items() if "expert" in x or "random" in x}
    
    color_map= {label: colors[i] for i, label in enumerate(data_dict)}
    max_count = 0
    for c,(label, data) in enumerate(data_dict.items()):
        # Plot overlapping histograms
        counts, _, _ =ax.hist(data, bins=10, alpha=0.5, label=label, color=color_map[label])
        max_count = max(max_count, max(counts))


    ax.set_ylim(0, max_count)  # Set the y-axis limit to max episodes
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # Add labels and legend

    ax.set_xlabel("Wall Hits")
    ax.set_ylabel("Num of Episodes")
    handles, labels = ax.get_legend_handles_labels()
    sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)
    ax.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(f"{args.env}_{s}_wall_hits_plot.png", bbox_inches='tight' )
    plt.close()





# plt.figure(figsize=(10, 6))
# for i, (label, data) in enumerate(wall_hits.items()):
#     plt.hist(data, bins=10, alpha=0.5, label=label, color=colors[i % len(colors)])

# # Set y-axis to use only integer ticks and reflect max episodes
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# plt.ylim(0, max_episodes)  # Set the y-axis limit to max episodes

# # Add labels and legend
# plt.xlabel("Wall Hits")
# plt.ylabel("Num of Episodes")


# handles, labels = plt.get_legend_handles_labels()
# sorted_handles_labels = sorted(zip(labels, handles), key=lambda x: x[0])
# sorted_labels, sorted_handles = zip(*sorted_handles_labels)
# plt.legend(sorted_handles, sorted_labels, loc='upper left', bbox_to_anchor=(1, 1))

# # Display the plot
# plt.savefig(f"{args.env}_wall_hits_plot.png", bbox_inches='tight' )
