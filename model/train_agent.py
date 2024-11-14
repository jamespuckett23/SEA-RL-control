# train_agent.py

import gym
import numpy as np
from single_sea_env import SingleSEAEnv
from DDPG import DDPG
import matplotlib.pyplot as plt
import pickle
import torch
import threading
from pynput import keyboard  # Use pynput for key detection

# Global variable to toggle visualization
visualization_enabled = False

def toggle_visualization(env):
    """Toggles visualization dynamically based on user input (press 'v' to toggle)."""
    def on_press(key):
        global visualization_enabled
        try:
            if key.char == 'v':  # Press 'v' to toggle visualization
                visualization_enabled = not visualization_enabled
                env.visualize = visualization_enabled  # Update environment visualization flag
                print(f"Visualization {'enabled' if visualization_enabled else 'disabled'}")
        except AttributeError:
            pass  # Handle special keys that don't have a char attribute

    # Listen for key presses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()  # Keeps the listener running

def main():
    global visualization_enabled

    # Create the environment with visualization initially off
    env = SingleSEAEnv(visualize=visualization_enabled, mse_threshold=0.01)
    eval_env = None  # If you have a separate evaluation environment

    # Start a separate thread to listen for visualization toggle key
    threading.Thread(target=toggle_visualization, args=(env,), daemon=True).start()

    # Define options with necessary parameters
    class Options:
        def __init__(self):
            self.gamma = 0.99        # Discount factor
            self.alpha = 0.001       # Learning rate
            self.epsilon = 0.5       # Starting epsilon for exploration
            self.epsilon_min = 0.01  # Minimum epsilon
            self.epsilon_decay = 0.999 # Decay rate for epsilon
            self.num_episodes = 5000 # Number of episodes to train
            self.steps = 1000        # Maximum steps per episode
            self.layers = [128, 128, 64]
            self.replay_memory_size = 500000
            self.batch_size = 64
            self.update_target_estimator_every = 500

    options = Options()

    # Create an instance of DDPG
    agent = DDPG(env, eval_env, options)
    agent.actor_critic.load_state_dict(torch.load("actor_critic_5000_from_good_version.pth"))
    agent.target_actor_critic.load_state_dict(torch.load("target_actor_5000_from_good_version.pth"))

    rewards = []
    smoothed_rewards = []  # To store smoothed rewards
    smoothing_window = 200  # Window size for moving average

    # Set up dynamic plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Rewards Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    reward_line, = ax.plot([], [], label='Episode Reward', alpha=0.5)
    smooth_line, = ax.plot([], [], label='Smoothed Reward (SMA)', color='orange')
    ax.legend()
    ax.grid(True)
    
    def compute_moving_average(data, window_size):
        if len(data) < window_size:
            return np.mean(data)  # If not enough data, just use the mean
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def update_plot(episode, rewards, smoothed_rewards):
        reward_line.set_xdata(range(1, episode + 1))
        reward_line.set_ydata(rewards)

        smooth_line.set_xdata(range(smoothing_window, episode + 1))
        smooth_line.set_ydata(smoothed_rewards)

        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)

    # Training loop
    for episode in range(options.num_episodes):
        print(f"Episode {episode + 1} started.")
        rewards.append(agent.train_episode())
        smoothed_rewards = compute_moving_average(rewards, smoothing_window)

        # Optionally, print progress
        if (episode + 1) % 10 == 0:
            torch.save(agent.actor_critic.state_dict(), "actor_critic_6000_from_good_version.pth")
            torch.save(agent.target_actor_critic.state_dict(), "target_actor_6000_from_good_version.pth")
        
        update_plot(episode + 1, rewards, smoothed_rewards)
        if len(rewards) > smoothing_window and smoothed_rewards[-1] > 5000:
            break

    # Keep the plot open after training
    plt.ioff()
    plt.savefig('rewards_over_episodes.png')  # Save the final plot
    plt.show()

    env.close()

if __name__ == "__main__":
    main()