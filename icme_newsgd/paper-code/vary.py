import numpy as np
import os
import matplotlib.pyplot as plt



# Function to calculate mean and variance for each round
def calculate_mean_variance_for_rounds(data_dir=None, num_rounds=200, num_batches=9):
    round_means = []
    round_variances = []

    for round_num in range(1, num_rounds + 1):
        batch_means = []
        batch_variances = []

        for batch_index in range(2, num_batches + 2):
            file_name = f"{round_num}_{batch_index}_tensor.npy"
            file_path = file_name

            # Load the .npy file and compute mean and variance
            if os.path.exists(file_path):
                data = np.load(file_path)
                batch_means.append(np.mean(data))
                batch_variances.append(np.var(data))

        # Calculate mean of means and variances for the current round
        if batch_means and batch_variances:
            round_mean = np.mean(batch_means)
            round_variance = np.mean(batch_variances)
            round_means.append(round_mean)
            round_variances.append(round_variance)

    return round_means, round_variances


# Function to plot and save the results
def plot_mean_variance(mean_values, variance_values, output_file='mean_variance_plot.png'):
    rounds = range(1, len(mean_values) + 1)

    plt.figure(figsize=(10, 5))

    # Plot mean and variance
    plt.plot(rounds, mean_values, label="Mean", marker='o')
    plt.plot(rounds, variance_values, label="Variance", marker='x')

    # Labels and title
    plt.xlabel("Rounds")
    plt.ylabel("Values")
    plt.title("Mean and Variance Across Rounds")
    plt.legend()

    # Save the plot
    plt.savefig(output_file)
    plt.close()


# Calculate the mean and variance for each round
round_means, round_variances = calculate_mean_variance_for_rounds()

# Plot and save the results
plot_mean_variance(round_means, round_variances, output_file='/code/newsgd/paper-code/cos/fig/mean_variance_plot.png')
