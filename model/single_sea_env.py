# single_sea_env.py

import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
from SingleSEA import SingleSEA
from scipy.integrate import solve_ivp


class SingleSEAEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, visualize=False, mse_threshold=0.01):
        super(SingleSEAEnv, self).__init__()

        # Initialize the SingleSEA system with default parameters
        params = {
            'J_m': 0.44,           # Motor inertia (kg·m²)
            'J_j': 0.29,           # Joint/load inertia (kg·m²)
            'K_s': 1180.0,         # Spring stiffness (N·m/rad)
            'B_m': 17.9,           # Motor damping (N·m·s/rad)
            'link_length': 1.0,    # Length of the link (m)
            'F': 0.0,              # Initial external force magnitude (N)
            'alpha': 0.0,          # Initial external force direction (rad)
        }

        self.system = SingleSEA(params)

        # Action space: Discrete actions representing torque from -350 to 350 N·m in increments of 1
        self.action_space = spaces.Discrete(701)  # Actions 0 to 700

        # Observation space: [theta_m, omega_m, theta_j, omega_j]
        high = np.array([np.pi, 20.0, np.pi, 20.0, np.pi, 350.0])
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # Initial state
        self.state = np.array([0.0, 0.0, 0.0, 0.0])

        # Time parameters
        self.dt = 0.005  # Time step
        self.time_elapsed = 0.0

        # Visualization flag
        self.visualize = visualize
        if self.visualize:
            self._setup_visualization()

        # Placeholder for desired state (will be set in reset)
        self.desired_state = np.array([0.0, 0.0])

        # Mean squared error threshold for episode termination
        self.mse_threshold = mse_threshold

        self.counter = 0.0
        self.F = np.random.uniform(0.0, 100.0)  # Force between 0 and 100 N
        self.alpha = np.random.uniform(-np.pi, np.pi)  # Direction between -π and π radians

        # Set to use torque commands instead of gains
        self.system.set_use_gains(False)

        self.L = np.array([20.0, 0.2, 50.0, 0.5])

        self.s = self.state.copy()

    def step(self, action):
        # Map the action index to torque value
        torque = action - 350  # Torque ranges from -350 to 350 N·m

        # Set the torque command
        self.system.set_ff_tau(torque)
        self.system.set_desired_state(self.desired_state)

        self.system.set_external_force(self.F, self.alpha)

        # Compute state derivatives
        # dxdt = self.system.dynamics(self.time_elapsed, self.state)
        t_curr = self.time_elapsed + self.dt
        # Integrate from t_prev to t_curr
        sol = solve_ivp(
            fun=self.system.dynamics,
            t_span=[self.time_elapsed, t_curr],
            y0=self.s,
            method='RK45',
            max_step=self.dt,  # Use dt_step here
            rtol=1e-5,
            atol=1e-8
        )

        # Update the state
        self.s = sol.y[:, -1]
        # Update state
        self.state = self.s
        print("Desired State: ", self.desired_state)
        print("Current State: ", self.state)
        self.time_elapsed += self.dt

        # Optionally render the environment
        if self.visualize:
            self.render(self.F, self.alpha)

        # Calculate the squared error between desired and current state
        position_error =  (self.desired_state[0] - self.state[0]) ** 2
        velocity_error = self.state[1] ** 2 + self.state[3] ** 2
        torque_error   =  (self.desired_state[1] - self.system.K_s * (self.state[2] - self.state[0]))**2
        state_error = 500.0 * position_error + velocity_error + 50.0 * torque_error
        # Penalize large motor torque (encourage energy efficiency)
        torque_penalty = 50.0 * (torque ** 2)
        torque_violation_penalty = 1000000.0 * max(0, abs(self.system.K_s * (self.state[2] - self.state[0])) - 350)
        squared_error = state_error ** 2
        mse = np.mean(squared_error)

        # Reward function
        # Negative of mean squared error to minimize it, minus a small time penalty
        time_penalty = 1000.0  # Adjust the time penalty coefficient as needed
        # reward = -1000.0*position_error - time_penalty * self.time_elapsed
        # reward = -1000.0*position_error - velocity_error - 10.0 * torque_error - torque_penalty - torque_violation_penalty
        reward = -100000.0*position_error 

        # Check if the mean squared error is below the threshold
        done = position_error <= self.mse_threshold
        if done:
            reward = 2000.0
        # Additional info (optional)
        info = {
            'mse': mse,
            'time_elapsed': self.time_elapsed,
            'external_force': self.F,
            'external_alpha': self.alpha
        }

        RL_state = np.concatenate((self.state, self.desired_state), axis=0)

        return RL_state.copy(), reward, done, info

    def reset(self):
        # Reset state and time
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time_elapsed = 0.0
        # self.F = np.random.uniform(0.0, 100.0)  # Force between 0 and 100 N
        # self.alpha = np.random.uniform(-np.pi, np.pi)  # Direction between -π and π radians
        self.F = 0.0  # Force between 0 and 100 N
        self.alpha = 0.0  # Direction between -π and π radians

        # Generate a new random desired state
        # Calculate maximum allowed difference between desired motor and joint angles
        delta_theta_max = self.system.max_torque / self.system.K_s  # 350 / 1180 ≈ 0.2966 rad

        # Generate random desired motor angle
        theta_m_desired = np.random.uniform(-np.pi, np.pi)

        # Generate desired torque measurement
        torque_desired = self.F * self.system.link_length * np.sin(self.alpha - theta_m_desired)

        # Generate desired velocities (set to zero for simplicity)
        omega_m_desired = 0.0
        omega_j_desired = 0.0

        # Set the desired state
        self.desired_state = np.array([theta_m_desired, torque_desired])

        # Reset visualization if enabled
        if self.visualize:
            self._reset_visualization()
        RL_state = np.concatenate((self.state, self.desired_state), axis=0)

        return RL_state.copy()

    def render(self, external_force=0.0, external_alpha=0.0, mode='human'):
        if self.visualize:
            self._update_visualization(external_force, external_alpha)

    def close(self):
        if self.visualize:
            self._close_visualization()

    # Visualization setup methods
    def _setup_visualization(self):
        self.fig = plt.figure(figsize=(24, 16))
        gs = self.fig.add_gridspec(3, 2)

        # Main animation axis
        self.ax_anim = self.fig.add_subplot(gs[0:2, 0])
        link_length = self.system.link_length
        self.ax_anim.set_xlim(-2 * link_length, 2 * link_length)
        self.ax_anim.set_ylim(-2 * link_length, 2 * link_length)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True)
        self.ax_anim.set_title('Single SEA Visualization with External Force')

        # Initialize plot elements
        self.base_position = np.array([0, 0])  # Base position
        self.line_j, = self.ax_anim.plot([], [], 'o-', lw=2, label='Joint Link')
        self.line_m, = self.ax_anim.plot([], [], 'r-', lw=2, label='Motor Link')
        self.disturbance_arrow, = self.ax_anim.plot([], [], 'g-', lw=2, label='External Force')
        self.time_text = self.ax_anim.text(0.02, 0.95, '', transform=self.ax_anim.transAxes)
        self.ax_anim.legend()

        # Show the plot
        plt.ion()
        plt.show()

    def _reset_visualization(self):
        self.ax_anim.cla()
        link_length = self.system.link_length
        self.ax_anim.set_xlim(-2 * link_length, 2 * link_length)
        self.ax_anim.set_ylim(-2 * link_length, 2 * link_length)
        self.ax_anim.set_aspect('equal')
        self.ax_anim.grid(True)
        self.ax_anim.set_title('Single SEA Visualization with External Force')
        self.line_j, = self.ax_anim.plot([], [], 'o-', lw=2, label='Joint Link')
        self.line_m, = self.ax_anim.plot([], [], 'r-', lw=2, label='Motor Link')
        self.disturbance_arrow, = self.ax_anim.plot([], [], 'g-', lw=2, label='External Force')
        self.time_text = self.ax_anim.text(0.02, 0.95, '', transform=self.ax_anim.transAxes)
        self.ax_anim.legend()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_visualization(self, external_force, external_alpha):
        import numpy as np

        theta_m = self.state[0]
        theta_j = self.state[2]

        # Update link lines
        link_length = self.system.link_length
        x_j = self.base_position[0] + link_length * np.cos(theta_j)
        y_j = self.base_position[1] + link_length * np.sin(theta_j)

        x_m = self.base_position[0] + link_length * np.cos(theta_m)
        y_m = self.base_position[1] + link_length * np.sin(theta_m)

        self.line_j.set_data([self.base_position[0], x_j], [self.base_position[1], y_j])
        self.line_m.set_data([self.base_position[0], x_m], [self.base_position[1], y_m])

        # Update disturbance arrow
        arrow_scale = 0.5  # Scale factor for the arrow length
        arrow_x = x_j
        arrow_y = y_j
        arrow_dx = arrow_scale * (external_force / 100.0) * np.cos(external_alpha)
        arrow_dy = arrow_scale * (external_force / 100.0) * np.sin(external_alpha)
        self.disturbance_arrow.set_data([arrow_x, arrow_x + arrow_dx], [arrow_y, arrow_y + arrow_dy])

        # Update time text
        self.time_text.set_text(f'Time = {self.time_elapsed:.2f} s')

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _close_visualization(self):
        plt.ioff()
        plt.close(self.fig)
