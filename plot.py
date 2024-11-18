import pandas as pd
import matplotlib.pyplot as plt

# Increase font sizes globally
plt.rcParams.update(
    {
        "font.size": 16,  # Base font size for all text
        "axes.titlesize": 20,  # Font size for axes titles
        "axes.labelsize": 18,  # Font size for axes labels
        "legend.fontsize": 16,  # Font size for legend
        "xtick.labelsize": 14,  # Font size for x-axis tick labels
        "ytick.labelsize": 14,  # Font size for y-axis tick labels
    }
)

# File path
file_path = "loss.csv"

# Read the CSV file
data = pd.read_csv(file_path)

# Apply smoothing to the values
smoothing_factor = 0.9
smoothed_values = data["value"].ewm(alpha=(1 - smoothing_factor)).mean()

# Calculate a range for the shaded area
# Using a simple deviation metric for demonstration
window_size = 20  # Number of steps to calculate a rolling window deviation
std_dev = data["value"].rolling(window=window_size).std()

# Plot the smoothed loss function with a shaded area
plt.figure(figsize=(9, 6))
plt.plot(data["step"], smoothed_values, label="Smoothed Loss", color="blue")
plt.fill_between(
    data["step"],
    smoothed_values - std_dev,
    smoothed_values + std_dev,
    color="blue",
    alpha=0.2,
    label="Variability",
)

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim(3, 7.5)  # Set y-axis limits
plt.legend()
plt.grid(True)
plt.show()
