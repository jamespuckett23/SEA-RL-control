# train_agent.py

import gym
import numpy as np
from single_sea_env import SingleSEAEnv
from DDPG import DDPG  # Replace 'your_module_name' with the actual module name
import matplotlib.pyplot as plt
import pickle
import torch


def main():
    
    # Create the environment
    env = SingleSEAEnv(visualize=False, mse_threshold=0.01)
    eval_env = None  # If you have a separate evaluation environment

    # Define options with necessary parameters
    class Options:
        def __init__(self):
            self.gamma = 0.99        # Discount factor
            self.alpha = 0.001         # Learning rate
            self.epsilon = 0.5       # Starting epsilon for exploration
            self.epsilon_min = 0.01    # Minimum epsilon
            self.epsilon_decay = 0.999 # Decay rate for epsilon            self.num_episodes = 5000 # Number of episodes to train
            self.num_episodes = 10000 # Number of episodes to train
            self.steps = 1000        # Maximum steps per episode
            self.layers = [128, 128, 64]
            self.replay_memory_size = 500000
            self.batch_size = 64
            self.update_target_estimator_every = 500

    options = Options()

    # Create an instance of ApproxQLearning
    agent = DDPG(env, eval_env, options)
    agent.actor_critic.load_state_dict(torch.load("actor_critic_test_5000.pth"))
    agent.target_actor_critic.load_state_dict(torch.load("target_actor_test_5000.pth"))

    rewards =[]

    smoothed_rewards = []  # To store smoothed rewards
    smoothing_window = 100  # Window size for moving average

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
            print(f"Episode {episode + 1} completed.")
            torch.save(agent.actor_critic.state_dict(), "actor_critic_no_done_10000.pth")
            torch.save(agent.target_actor_critic.state_dict(), "target_actor_no_done_10000.pth")
            print(f"Model saved at episode {episode}")
        
        update_plot(episode + 1, rewards, smoothed_rewards)

        # Keep the plot open after training
    plt.ioff()
    plt.savefig('rewards over episodes.png')  # Save the final plot
    plt.show()

    env.close()

if __name__ == "__main__":
    main()
