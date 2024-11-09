import matplotlib.pyplot as plt
import matplotlib 

matplotlib.use("Agg")
plt.ioff()

def exponential_smoothing(data, alpha=0.1):
    """Compute exponential smoothing."""
    smoothed = [data[0]]  # Initialize with the first data point
    for i in range(1, len(data)):
        st = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(st)
    return smoothed

def live_plot(data_dict, plot_update_counter, alpha=0.1):
    """Plot the live graph with multiple subplots, without IPython dependencies."""

    plt.style.use('ggplot')
    n_plots = len(data_dict)
    # fig, axes = plt.subplots(nrows=n_plots, figsize=(10, 20 * n_plots), squeeze=False)
    n_cols = 3  # Number of columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate rows needed based on columns

    # Create the figure with adjusted size
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 7 * n_rows), squeeze=False)

    # Adjust spacing between plots
    plt.subplots_adjust(hspace=1, wspace=0.3)
    plt.ion()  # Interactive mode on to allow live updates

    while plt.fignum_exists(fig.number):  # Infinite loop to simulate live data update
        if plot_update_counter[0] % 1 ==0:
            for ax, (label, data) in zip(axes.flatten(), data_dict.items()):
                ax.clear()
                if data:
                    data_copy = data.copy()
                    # Compute and plot smoothed values
                    ma = exponential_smoothing(data_copy, alpha)
                    ax.plot(data_copy, label=label, color="yellow", linestyle='--')
                    ma_idx_start = len(data_copy) - len(ma)
                    ax.plot(range(ma_idx_start, len(data_copy)), ma, label="Smoothed Value",
                            linestyle="-", color="purple", linewidth=2)
                    ax.relim()
                    ax.autoscale_view()
                # ax.set_ylabel(label)
                ax.set_title(label)
                ax.legend(loc='upper right')
            


        plt.draw()  # Redraw the current figure
        plt.pause(0.5)  # Pause to update the plot


    plt.ioff()  # Turn off interactive mode when the loop exits
    if plot_update_counter[0] == 50000-1:
        fig.savefig("Training loss.png")
    plt.close(fig)  # Close the figure window gracefully



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
    plt.close()