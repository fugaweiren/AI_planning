import pickle
import matplotlib.pyplot as plt
from plot import live_plot, exponential_smoothing

# # Load the dictionary from the file
# with open('experiment1.pickle', 'rb') as handle:
#     data_to_plot = pickle.load(handle)

# with open('D:\Courses\CS5446 AI Planning & Decision Making\experiment1.pickle', 'rb') as handle:
#     data_to_plot = pickle.load(handle)

# Print the loaded dictionary to verify

def plot_final_results(data_dict, alpha=0.1, save_path='final_plot.png'):
    plt.style.use('ggplot')
    n_plots = len(data_dict)
    n_cols = 3  # Number of columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed based on columns
    # Create the figure with adjusted sizeè
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7* n_cols, 5* n_rows), squeeze=True)
    # Adjust spacing between plots
    plt.subplots_adjust(hspace=1, wspace=0.3)

    for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
        print(f"Plotting {label}, data length: {len(data)}")  # Debugging print statement
        if data:
            if label == "Total Reward" and len(data) > 1e7:
                data = data[::1000]

            ax.plot(data, label=label, color="yellow", linestyle='--')
            # Compute and plot smoothed values
            ma = exponential_smoothing(data, alpha)
            ma_idx_start = len(data) - len(ma)
            ax.plot(range(ma_idx_start, len(data)), ma, label="Smoothed Value",
                    linestyle="-", color="purple", linewidth=2)
            ax.relim()
            ax.autoscale_view()
        ax.set_title(label)
        ax.legend(loc='upper right')
    plt.savefig(save_path)
    # plt.show()
# plot_final_results(data_to_plot)