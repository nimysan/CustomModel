#!/usr/bin/env python3
"""
Script to create detailed visualizations of training metrics from step_wise_training_metrics.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('step_wise_training_metrics.csv')

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

# Plot 1: Training loss vs step number (as before)
ax1.plot(df['step_number'], df['training_loss'], marker='o', linestyle='-', color='blue', label='Training Loss')

# Add markers for each epoch
epochs = df['epoch_number'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

for i, epoch in enumerate(epochs):
    epoch_data = df[df['epoch_number'] == epoch]
    ax1.scatter(epoch_data['step_number'], epoch_data['training_loss'], 
                color=colors[i], s=100, label=f'Epoch {epoch}', zorder=5)

# Add vertical lines to separate epochs
for epoch in epochs[1:]:
    first_step = df[df['epoch_number'] == epoch]['step_number'].min()
    ax1.axvline(x=first_step-0.5, color='gray', linestyle='--', alpha=0.7)

# Add labels and title for first plot
ax1.set_xlabel('Step Number')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss vs Step Number')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# Add epoch labels at the top
for i, epoch in enumerate(epochs):
    epoch_data = df[df['epoch_number'] == epoch]
    mid_x = (epoch_data['step_number'].min() + epoch_data['step_number'].max()) / 2
    ax1.text(mid_x, ax1.get_ylim()[1] * 1.01, f'Epoch {epoch}', 
             horizontalalignment='center', fontsize=12)

# Plot 2: Loss by epoch (average and min)
epoch_avg_loss = df.groupby('epoch_number')['training_loss'].mean()
epoch_min_loss = df.groupby('epoch_number')['training_loss'].min()
epoch_max_loss = df.groupby('epoch_number')['training_loss'].max()

x = np.arange(len(epochs))
width = 0.25

# Plot bars for average, min, and max loss by epoch
ax2.bar(x - width, epoch_avg_loss, width, label='Average Loss', color='skyblue')
ax2.bar(x, epoch_min_loss, width, label='Min Loss', color='green')
ax2.bar(x + width, epoch_max_loss, width, label='Max Loss', color='salmon')

# Add labels and title for second plot
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Statistics by Epoch')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Epoch {epoch}' for epoch in epochs])
ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
ax2.legend()

# Add value labels on top of bars
for i, v in enumerate(epoch_avg_loss):
    ax2.text(i - width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(epoch_min_loss):
    ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
for i, v in enumerate(epoch_max_loss):
    ax2.text(i + width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig('training_loss_detailed_plot.png', dpi=300)
print("Detailed plot saved as training_loss_detailed_plot.png")

# Show the plot
plt.show()
