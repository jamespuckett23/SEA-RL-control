# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) inform Guni Sharon at 
# guni@tamu.edu regarding your usage (relevant statistics is reported to NSF).
# The development of this assignment was supported by NSF (IIS-2238979).
# Contributors:
# The core code base was developed by Guni Sharon (guni@tamu.edu).
# The PyTorch code was developed by Sheelabhadra Dey (sheelabhadra@tamu.edu).

import random
from copy import deepcopy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from Abstract_Solver import AbstractSolver

class QFunction(nn.Module):
    """
    Q-network definition.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
    ):
        super().__init__()
        sizes = [obs_dim] + hidden_sizes + [act_dim]
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))

    def forward(self, obs):
        x = torch.cat([obs], dim=-1)
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x).squeeze(dim=-1)


class DQN(AbstractSolver):
    def __init__(self, env, eval_env, options, model_path=None):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        # Check if a GPU is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Create Q-network
        self.model = QFunction(
            env.observation_space.shape[0],
            env.action_space.n,
            self.options.layers,
        ).to(self.device)
        # Create target Q-network
        self.target_model = deepcopy(self.model).to(self.device)

        if model_path:
            print(f"Loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set the model to evaluation mode

        # Set up the optimizer
        self.optimizer = AdamW(
            self.model.parameters(), lr=self.options.alpha, amsgrad=True
        )
        # Define the loss function
        self.loss_fn = nn.SmoothL1Loss()

        # Freeze target network parameters
        for p in self.target_model.parameters():
            p.requires_grad = False

        # Replay buffer
        self.replay_memory = deque(maxlen=options.replay_memory_size)

        # Number of training steps so far
        self.n_steps = 0

        self.counter = 0

    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.load_state_dict(self.model.state_dict())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            The probabilities (as a Numpy array) associated with each action for 'state'.

        Use:
            self.env.action_space.n: Number of avilable actions
            self.torch.as_tensor(state): Convert Numpy array ('state') to a tensor
            self.model(state): Returns the predicted Q values at a 
                'state' as a tensor. One value per action.
            torch.argmax(values): Returns the index corresponding to the highest value in
                'values' (a tensor)
        """
        # Don't forget to convert the states to torch tensors to pass them through the network.
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        probs = np.zeros(self.env.action_space.n)
        state_tensor = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        values = self.model(state_tensor)
        best_action = torch.argmax(values)
        epsilon = self.options.epsilon
        for action in range(self.env.action_space.n):
            if action == best_action:
                probs[action] = 1-epsilon + epsilon/self.env.action_space.n
            else:
                probs[action] = epsilon/self.env.action_space.n
        return probs



    def compute_target_values(self, next_states, rewards, dones):
        """
        Computes the target q values.

        Returns:
            The target q value (as a tensor) of shape [len(next_states)]
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        gamma = self.options.gamma
        num_steps = len(next_states)
        q = np.zeros(num_steps)
        for j in range(num_steps):
            if dones[j]:
                q[j] = rewards[j]
            else:
                state_tensor = torch.as_tensor(next_states[j], dtype=torch.float32).to(self.device)
                values = self.target_model(state_tensor)
                best_action = torch.argmax(values)
                q_hat = gamma * values[best_action]
                q[j] = rewards[j] + q_hat
        return torch.as_tensor(q).to(self.device)



    def replay(self):
        """
        TD learning for q values on past transitions.

        Use:
            self.target_model(state): predicted q values as an array with entry
                per action
        """
        if len(self.replay_memory) > self.options.batch_size:
            minibatch = random.sample(self.replay_memory, self.options.batch_size)
            minibatch = [
                np.array(
                    [
                         transition[idx].cpu().numpy() if isinstance(transition[idx], torch.Tensor) else transition[idx]
                    for transition, idx in zip(minibatch, [i] * len(minibatch))
                    ]
                )
                for i in range(5)
            ]
            states, actions, rewards, next_states, dones = minibatch
            # Convert numpy arrays to torch tensors
            states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
            actions = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
            rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
            next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
            dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

            # Current Q-values
            current_q = self.model(states)
            # Q-values for actions in the replay memory
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            # Calculate loss
            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation. Finds the optimal greedy policy
        while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return probabilities of actions.
            np.random.choice(array, p=prob): sample an element from 'array' based on their corresponding
                probabilites 'prob'.
            self.memorize(state, action, reward, next_state, done): store the transition in the replay buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps (HINT: to be done across episodes)
        """

        # Reset the environment
        state = torch.as_tensor(self.env.reset(), dtype=torch.float32).to(self.device)
        reward = 0.0
        print(state)

        for i in range(self.options.steps):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            self.n_steps += 1
            probs = self.epsilon_greedy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = self.step(action)

            next_state = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
            reward = torch.as_tensor(reward, dtype=torch.float32).to(self.device)

            self.memorize(state, action, reward, next_state, done)

            self.replay()
            if done:
                print("Exited with reward: ", reward, "  Number of steps: ", i)
                break
            
            state = next_state
            
            if self.n_steps % self.options.update_target_estimator_every == 0:
                self.update_target_model()
        self.decay_epsilon()
        if self.counter % 10 == 0:
            torch.save(self.model.state_dict(), f"model_episode_{self.counter}.pth")
            print(f"Model saved for episode {self.counter}")
        self.counter += 1
        print("Final Reward: ", reward)


    def __str__(self):
        return "DQN"

    def plot(self, stats, smoothing_window, final=False):
        plotting.plot_episode_stats(stats, smoothing_window, final=final)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            state = torch.as_tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            return torch.argmax(q_values).detach().numpy()

        return policy_fn
    def decay_epsilon(self):
        """
        Decays epsilon after each episode.
        """
        self.options.epsilon = max(
            self.options.epsilon_min,
            self.options.epsilon * self.options.epsilon_decay
    )
