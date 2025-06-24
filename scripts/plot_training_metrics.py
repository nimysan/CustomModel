#!/usr/bin/env python3
"""
Script to visualize training metrics from step_wise_training_metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('step_wise_training_metrics.csv')

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot training loss vs step number
plt.plot(df['step_number'], df['training_loss'], marker='o', linestyle='-', color='blue', label='Training Loss')

# Add markers for each epoch
epochs = df['epoch_number'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

for i, epoch in enumerate(epochs):
    epoch_data = df[df['epoch_number'] == epoch]
    plt.scatter(epoch_data['step_number'], epoch_data['training_loss'], 
                color=colors[i], s=100, label=f'Epoch {epoch}', zorder=5)

# Add vertical lines to separate epochs
for epoch in epochs[1:]:
    first_step = df[df['epoch_number'] == epoch]['step_number'].min()
    plt.axvline(x=first_step-0.5, color='gray', linestyle='--', alpha=0.7)

# Add labels and title
plt.xlabel('Step Number')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Step Number')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Add epoch labels at the top
for i, epoch in enumerate(epochs):
    epoch_data = df[df['epoch_number'] == epoch]
    mid_x = (epoch_data['step_number'].min() + epoch_data['step_number'].max()) / 2
    plt.text(mid_x, plt.ylim()[1] * 1.01, f'Epoch {epoch}', 
             horizontalalignment='center', fontsize=12)

# Add annotations for min loss in each epoch
for epoch in epochs:
    epoch_data = df[df['epoch_number'] == epoch]
    min_loss_row = epoch_data.loc[epoch_data['training_loss'].idxmin()]
    plt.annotate(f"{min_loss_row['training_loss']:.3f}",
                 (min_loss_row['step_number'], min_loss_row['training_loss']),
                 textcoords="offset points", xytext=(0,-15), ha='center')

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig('training_loss_plot.png', dpi=300)
print("Plot saved as training_loss_plot.png")

# Show the plot
plt.show()
