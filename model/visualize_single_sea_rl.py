import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from single_sea_env import SingleSEAEnv
from SingleSEA import SingleSEA
import torch
from DDPG import DDPG  # Replace 'your_module_name' with the actual module name


def main():
    # Simulation parameters
    t_start = 0.0
    t_end = 100.0
    dt = 0.001
    t = np.arange(t_start, t_end + dt, dt)

    # Load trained AI model
    env = SingleSEAEnv(visualize=False, mse_threshold=0.01)
    # Define options with necessary parameters
    class Options:
        def __init__(self):
            self.gamma = 0.99        # Discount factor
            self.alpha = 0.001         # Learning rate
            self.epsilon = 0.5       # Starting epsilon for exploration
            self.epsilon_min = 0.01    # Minimum epsilon
            self.epsilon_decay = 0.999 # Decay rate for epsilon            self.num_episodes = 5000 # Number of episodes to train
            self.num_episodes = 5000 # Number of episodes to train
            self.steps = 1000        # Maximum steps per episode
            self.layers = [128, 128, 64]
            self.replay_memory_size = 500000
            self.batch_size = 64
            self.update_target_estimator_every = 500

    options = Options()
    actor_model = DDPG(env, None, options)
    actor_model.actor_critic.load_state_dict(torch.load("actor_critic_test_5000.pth"))
    actor_model.target_actor_critic.load_state_dict(torch.load("target_actor_test_5000.pth"))
    # Single SEA parameters
    params = {
        'J_m': 0.44,
        'J_j': 0.29,
        'K_s': 1180.0,
        'B_m': 17.9,
        'link_length': 1.0,
        'F': 0.0,
        'alpha': 0.0,
        'K1': 0.0,
        'B1': 0.0,
        'K2': 0.0,
        'B2': 0.0
    }

    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # Initial conditions
    link_length = params['link_length']
    base_position = np.array([0, 0])

    # Visualization setup
    fig, ax = plt.subplots()
    ax.set_xlim(-2 * link_length, 2 * link_length)
    ax.set_ylim(-2 * link_length, 2 * link_length)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Single SEA Visualization with AI Controller')

    # Link and disturbance elements
    line_j, = ax.plot([], [], 'o-', lw=2, label='Joint Link')
    line_m, = ax.plot([], [], 'r-', lw=2, label='Motor Link')
    disturbance_arrow, = ax.plot([], [], 'g-', lw=2, label='Disturbance Force')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend()

    # Slider setup
    axcolor = 'lightgoldenrodyellow'
    ax_dist_mag = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
    ax_dist_dir = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)
    ax_pos_des = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)

    slider_dist_mag = Slider(ax_dist_mag, 'Dist. Magnitude', 0.0, 100.0, valinit=0.0)
    slider_dist_dir = Slider(ax_dist_dir, 'Dist. Direction', -np.pi, np.pi, valinit=0.0)
    slider_pos_des = Slider(ax_pos_des, 'Position Setpoint', -np.pi, np.pi, valinit=0.1)

    # Initialize disturbance parameters
    disturbance_magnitude = slider_dist_mag.val
    disturbance_direction = slider_dist_dir.val
    desired_position = slider_pos_des.val

    def update_dist_mag(val):
        nonlocal disturbance_magnitude
        disturbance_magnitude = val

    def update_dist_dir(val):
        nonlocal disturbance_direction
        disturbance_direction = val

    def update_pos_des(val):
        nonlocal desired_position
        desired_position = val

    slider_dist_mag.on_changed(update_dist_mag)
    slider_dist_dir.on_changed(update_dist_dir)
    slider_pos_des.on_changed(update_pos_des)

    state = np.concatenate([x0, [desired_position, 0.0]])  # Initial state with placeholders for desired position and torque
    t_prev = t[0]

    def init():
        nonlocal state
        # Compute initial desired torque
        torque_desired = disturbance_magnitude * env.system.link_length * np.sin(
            disturbance_direction - desired_position
        )
        state[4:] = [desired_position, torque_desired]  # Update the desired state in the initial condition
        line_j.set_data([], [])
        line_m.set_data([], [])
        disturbance_arrow.set_data([], [])
        time_text.set_text('')
        return line_j, line_m, disturbance_arrow, time_text

    def update(frame):
        nonlocal state, t_prev

        t_curr = t[frame]
        env.system.set_external_force(disturbance_magnitude, disturbance_direction)

        # Update desired torque based on the slider values
        torque_desired = disturbance_magnitude * env.system.link_length * np.sin(
            disturbance_direction - desired_position
        )
        state[4:] = [desired_position, torque_desired]

        # AI-based torque action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            torque_action = actor_model.select_action(state_tensor)
            next_state, _, _, _ = actor_model.step(torque_action)

        env.system.set_ff_tau(torque_action[0])
        state = next_state
        t_prev = t_curr

        # Visual updates
        theta_m, _, theta_j, _ = state[:4]
        x_j = base_position[0] + link_length * np.cos(theta_j)
        y_j = base_position[1] + link_length * np.sin(theta_j)
        x_m = base_position[0] + link_length * np.cos(theta_m)
        y_m = base_position[1] + link_length * np.sin(theta_m)

        line_j.set_data([base_position[0], x_j], [base_position[1], y_j])
        line_m.set_data([base_position[0], x_m], [base_position[1], y_m])

        arrow_scale = 0.5
        arrow_x = x_j
        arrow_y = y_j
        arrow_dx = arrow_scale * (disturbance_magnitude / 100.0) * np.cos(disturbance_direction)
        arrow_dy = arrow_scale * (disturbance_magnitude / 100.0) * np.sin(disturbance_direction)
        disturbance_arrow.set_data([arrow_x, arrow_x + arrow_dx], [arrow_y, arrow_y + arrow_dy])

        time_text.set_text(f'Time = {t_curr:.2f}s')
        return line_j, line_m, disturbance_arrow, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=dt, blit=False, repeat=False)
    plt.show()

if __name__ == '__main__':
    main()
