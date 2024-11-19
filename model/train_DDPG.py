import gymnasium as gym
import numpy as np
from DDPG import DDPGAgent
from single_sea_env import SingleSEAEnv
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('Agg')

def load_latest_checkpoint(agent, path="./"):
    checkpoint_files = [f for f in os.listdir(path) if f.startswith("ddpg_models_episode")]
    
    # If there are any checkpoint files, find the latest one
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with open(os.path.join(path, latest_checkpoint), "rb") as f:
            agent = pickle.load(f)
        # Extract episode number from filename and continue from the next episode
        start_episode = int(latest_checkpoint.split('_')[-1].split('.')[0]) + 1
        print(f"Resuming from episode {start_episode} (loaded {latest_checkpoint})")
    else:
        start_episode = 0
        print("No checkpoints found. Starting training from scratch.")
    
    return agent, start_episode

# Initialize your custom Gym environment
env = SingleSEAEnv(visualize=False, mse_threshold=0.01)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_limit = env.action_space.high[0]  # Assuming symmetric action bounds

# Initialize DDPG agent
agent = DDPGAgent(state_dim, action_dim, action_limit)

rewards = []

# Training parameters
num_episodes = 2000
max_steps = 4000  # Maximum steps per episode
noise_scale = 0.1  # Initial noise scale

# Load the latest checkpoint if available
# agent, start_episode = load_latest_checkpoint(agent)
start_episode = 0

# Training loop
for episode in range(start_episode, num_episodes):
    state = env.reset()
    agent.noise.reset()  # Reset noise for each episode
    episode_reward = 0
    
    for step in range(max_steps):
        # Select action from the agent
        action = agent.act(state, noise_scale=noise_scale)

        # Step the environment with the selected action
        next_state, reward, done, info = env.step(action)
        
        print("state: ", state)
        print("action (torque): ", action)
        print("next_state: ", next_state)
        if done:
            print("finished")

        # Store experience in replay buffer
        agent.step(state, action, reward, next_state, done)
        
        # Move to the next state
        state = next_state
        episode_reward += reward
        
        # Check if the episode has ended
        if done:
            break
    
    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
    rewards.append(episode_reward)
    
    # Decay noise scale over time for less exploration as training progresses
    noise_scale = max(noise_scale * 0.99, 0.01)  # Gradually reduce noise

    # Save model every 100 episodes
    if episode % 100 == 0 and episode != 0:
        print("Saving models at episode:", episode)
        with open(f"ddpg_models_episode_{episode}.pkl", "wb") as f:
            pickle.dump(agent, f)  # Save the entire agent

# Plotting the rewards
plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.xscale('log')
plt.ylabel('Reward')
plt.title('Rewards over Episodes')
plt.grid(True)
plt.legend()
plt.savefig("rewards_plot.png")  # Save the plot as a file

# Close the environment
env.close()
