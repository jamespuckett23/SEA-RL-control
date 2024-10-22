# train_agent.py

import gym
import numpy as np
from single_sea_env import SingleSEAEnv
from DQN import DQN  # Replace 'your_module_name' with the actual module name
import matplotlib.pyplot as plt
import pickle

def main():
    # Create the environment
    env = SingleSEAEnv(visualize=True, mse_threshold=0.01)
    eval_env = None  # If you have a separate evaluation environment

    # Define options with necessary parameters
    class Options:
        def __init__(self):
            self.gamma = 0.99        # Discount factor
            self.alpha = 0.1         # Learning rate
            self.epsilon = 0.1       # Epsilon for epsilon-greedy policy
            self.num_episodes = 1000 # Number of episodes to train
            self.steps = 1000        # Maximum steps per episode
            self.layers = [32,32]
            self.replay_memory_size = 5000
            self.batch_size = 32
            self.update_target_estimator_every = 1000

    options = Options()

    # Create an instance of ApproxQLearning
    agent = DQN(env, eval_env, options, model_path=None)

    # Training loop
    for episode in range(options.num_episodes):
        print(f"Episode {episode + 1} started.")
        agent.train_episode()

        # Optionally, print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} completed.")
        # At the end of your training loop, save the models of the estimator
        # print("Saved Model")
        # with open("estimator_models.pkl", "wb") as f:
        #     pickle.dump(agent.estimator.models, f)  # Save the models (one per action)

    # After training, create a greedy policy and evaluate it
    policy = agent.create_greedy_policy()

    # Run the policy in the environment
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

        # Optionally render the environment
        # env.render()

    print(f"Total reward obtained: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
