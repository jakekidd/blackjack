import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot(
    rewards: List[int],
    wins: int,
    losses: int,
    draws: int,
    rolling_window: int = 100,
    max_points: int = 1000
):
    """
    Display rewards per episode and win/loss/draw breakdown in one window.

    Args:
        rewards (List[int]): List of rewards per episode.
        wins (int): Number of games won.
        losses (int): Number of games lost.
        draws (int): Number of games drawn.
        rolling_window (int): Window size for calculating the rolling average.
        max_points (int): Maximum number of points to display in the rewards graph.
    """
    # Enable dark mode
    plt.style.use('dark_background')

    # Downsample rewards if the data size exceeds max_points
    if len(rewards) > max_points:
        step = len(rewards) // max_points
        rewards = rewards[::step]
        x_axis = np.arange(0, len(rewards) * step, step)
    else:
        x_axis = np.arange(len(rewards))

    # Calculate the rolling average
    rolling_avg = np.convolve(rewards, np.ones(rolling_window) / rolling_window, mode='valid')
    rolling_avg_x = x_axis[:len(rolling_avg)]  # Adjust x-axis for the rolling average

    # Create a single figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Rewards over episodes
    axes[0].plot(x_axis, rewards, label="Total Reward", alpha=0.3, color='cyan')
    axes[0].plot(rolling_avg_x, rolling_avg, label=f"{rolling_window}-Episode Rolling Average", color='orange', linewidth=2)
    axes[0].set_title("Rewards per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(0, color='white', linestyle='--', linewidth=1)  # Baseline
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Win/Loss/Draw breakdown
    labels = ['Wins', 'Losses', 'Draws']
    counts = [wins, losses, draws]
    bars = axes[1].bar(labels, counts, color=['green', 'red', 'blue'], alpha=0.7)
    axes[1].set_title("Win/Loss/Draw Breakdown")
    axes[1].set_ylabel("Count")

    # Add data labels to bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, height + 0.5, str(height), ha='center', fontsize=12)

    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
