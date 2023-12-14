import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV files
file1 = "benchmark/logs/medusa_3h.csv"
file2 = "benchmark/logs/hydra_1l.csv"
file3 = "benchmark/logs/hydra_2l.csv"

data1 = pd.read_csv(file1, delimiter=',')
data2 = pd.read_csv(file2, delimiter=',')
data3 = pd.read_csv(file3, delimiter=',')

# Extract the data columns
maximum_steps = 1000
x1 = data1['Step'][:maximum_steps]
y1 = data1['test - train/loss'][:maximum_steps]

x2 = data2['Step'][:maximum_steps]
y2 = data2['test - train/loss'][:maximum_steps]

x3 = data3['Step'][:maximum_steps]
y3 = data3['test - train/loss'][:maximum_steps]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Create the plot
plt.figure(figsize=(10, 6))

plt.rcParams.update({'font.size': 25})

# Plot the training loss curves for Medusa and Hydra
plt.plot(x1[4:], moving_average(y1, 5), label='Medusa - 3 Heads')
plt.plot(x2[4:], moving_average(y2, 5), label='Hydra - 3 Heads - 1 Layer(s)')
plt.plot(x3[4:], moving_average(y3, 5), label='Hydra - 3 Heads - 2 Layer(s)')

# Customize the plot
plt.xlabel('Steps')
plt.ylabel('Training Loss')
plt.title('Training Loss Curves for Medusa and Hydra')
plt.legend()
plt.grid(True)

# Show the plot
# plt.show()
plt.savefig("benchmark/logs/training_loss.pdf", format="pdf", bbox_inches="tight")
