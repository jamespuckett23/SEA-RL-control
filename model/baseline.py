import numpy as np
import matplotlib.pyplot as plt
from single_sea_env import SingleSEAEnv
from SingleSEA import SingleSEA
import torch

class FSFController:
    def __init__(self, sea_system):
        self.sea_system = sea_system

    def get_action(self, state):
        # Using the fsf_controller method from the SingleSEA class
        motor_position, motor_velocity, spring_position, spring_velocity = state[:4]
        desired_position, desired_torque = state[4:]
        x = np.array([motor_position, motor_velocity, spring_position, spring_velocity])
        x_des = np.array([desired_position, 0.0, 0.0, 0.0])
        torque_action = self.sea_system.fsf_controller(x, x_des)
        return torque_action

def main():
    # Create the environment
    env = SingleSEAEnv(visualize=False, mse_threshold=0.01)
    sea_system = SingleSEA(env.params)
    controller = FSFController(sea_system)

    num_episodes = 5000
    max_steps = 200

    # Reward storage
    rewards_baseline = []

    plt.figure(figsize=(10, 6))

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = controller.get_action(state)
            next_state, reward, done, _ = env.step([action])
            total_reward += reward

            if done:
                break

            state = next_state

        rewards_baseline.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        # Plotting rewards as episodes complete
        plt.clf()
        plt.plot(rewards_baseline, label='FSF Controller', color='blue')
        avg_reward = np.mean(rewards_baseline)
        plt.axhline(y=avg_reward, color='red', linestyle='--', label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('FSF Controller Rewards Over Episodes')
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)

    # Save the final plot
    plt.savefig('baseline.png')
    plt.show()

if __name__ == "__main__":
    main()
