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
        self.params = {
            'J_m': 0.44,           # Motor inertia (kg·m²)
            'J_j': 0.29,           # Joint/load inertia (kg·m²)
            'K_s': 1180.0,         # Spring stiffness (N·m/rad)
            'B_m': 17.9,           # Motor damping (N·m·s/rad)
            'link_length': 1.0,    # Length of the link (m)
            'F': 0.0,              # Initial external force magnitude (N)
            'alpha': 0.0,          # Initial external force direction (rad)
            'K1': 694.8,
            'B1': 13.87,
            'K2': -10.0,
            'B2': -5.0,
        }

        self.system = SingleSEA(self.params)

        # Action space: Discrete actions representing torque from -100 to 100 N·m in increments of 1
        self.action_space = spaces.Box(
            low=-100.0,
            high=100.0,
            dtype=np.float32
        )


        # Observation space: [theta_m, omega_m, theta_j, omega_j]
        high = np.array([np.pi, 5.0, np.pi, 5.0, np.pi, 100.0])
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        # Initial state
        self.state = np.array(self.observation_space.sample()[:4])
        self.state[2] = self.state[0]
        self.state[1] = 0.0
        self.state[3] = self.state[1]

        # Time parameters
        self.dt = 0.01  # Time step
        self.time_elapsed = 0.0
        self.previous_position_error = 0
        self.time_in_zone = 0
        self.base_position = np.array([0, 0])  # Default base position

        # Visualization flag
        self.visualize = visualize
        self._setup_visualization()

        # Placeholder for desired state (will be set in reset)
        self.desired_state = np.array([0.0, 0.0])

        # Mean squared error threshold for episode termination
        self.mse_threshold = mse_threshold

        self.counter = 0.0
        self.F = np.random.uniform(0.0, 80.0)  # Force between 0 and 100 N
        self.alpha = np.random.uniform(-np.pi, np.pi)  # Direction between -π and π radians

        self.L = np.array([20.0, 0.2, 50.0, 0.5])

        self.s = self.state.copy()

    def step(self, action):
        # Map the action index to torque value
        torque = action[0]  # Torque ranges from -100 to 100 N·m

        # Set the torque command
        self.system.set_ff_tau(torque)
        state_des = [self.desired_state[0], 0.0, self.desired_state[1], 0.0]
        self.system.set_desired_state(state_des)

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
        self.time_elapsed += self.dt

        # Optionally render the environment
        if self.visualize:
            self.render(self.F, self.alpha)

        # Calculate the squared error between desired and current state
        position_error =  abs((self.desired_state[0] - self.state[0])) / 6.28
        velocity_error = (abs(self.state[1]) + abs(self.state[3])) / 20.0
        torque_error   =  abs((self.desired_state[1] - self.system.K_s * (self.state[2] - self.state[0])))/200.0
        # Penalize large motor torque (encourage energy efficiency)
        torque_penalty =  abs(torque) / 100.0
        torque_violation_penalty =  max(0, abs(self.system.K_s * (self.state[2] - self.state[0])) - 150)
        time_penalty = self.time_elapsed

        # Reward function
        reward = -10000.0*position_error - 25.0*torque_penalty - 10.0*time_penalty - 10.0*velocity_error - 2.0*torque_error 

        if position_error < 0.1:
            reward += 50.0

        if position_error < 0.01:
            reward += 100.0

        if position_error < 0.005:
            reward += 400.0
        
        if position_error < 0.005:
            self.time_in_zone += 1
        else:
            self.time_in_zone = 0
        if self.time_in_zone > 1:
            reward += 100 * self.time_in_zone

        if position_error > 0.8:
            reward -= 1000.0  # Soft penalty near violation

        if max(0, abs(self.system.K_s * (self.state[2] - self.state[0])) - 100) > 0:
            reward -= 1000.0

        if self.time_in_zone > 50:
            reward += 2000.0
            done = True
        elif position_error > 1.0:
            reward -= 5000.0
            print("POSITION VIOLATION")
            done = True
        elif torque_violation_penalty > 0:
            reward -= 5000.0
            print("TORQUE VIOLATION")
            done = True
        else:
            done = False
        # Additional info (optional)
        info = {
            'time_elapsed': self.time_elapsed,
            'external_force': self.F,
            'external_alpha': self.alpha
        }

        RL_state = np.concatenate((self.state, self.desired_state), axis=0)

        return RL_state.copy(), reward, done, info

    def reset(self):
        self.system = SingleSEA(self.params)
        # Set to use torque commands instead of gains        self.state[2] = self.state[0]
        self.state[1] = 0.0
        self.state[3] = self.state[1]
        self.system.set_use_gains(False)
        # Reset state and time
        self.state = np.array(self.observation_space.sample()[:4])
        self.state[2] = self.state[0]
        self.state[1] = 0.0
        self.state[3] = self.state[1]
        self.s = self.state.copy()
        self.time_elapsed = 0.0
        self.time_in_zone = 0.0
        self.F = np.random.uniform(0.0, 80.0)  # Force between 0 and 100 N
        self.alpha = np.random.uniform(-np.pi, np.pi)  # Direction between -π and π radians
        # self.F = 0.0  # Force between 0 and 100 N
        # self.alpha = 0.0  # Direction between -π and π radians

        # Generate a new random desired state
        # Calculate maximum allowed difference between desired motor and joint angles
        delta_theta_max = self.system.max_torque / self.system.K_s  # 100 / 1180 ≈ 0.2966 rad

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

        return RL_state.copy(), "none"

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
