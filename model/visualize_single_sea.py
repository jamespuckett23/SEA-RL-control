# interactive_visualize_single_sea.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from SingleSEA import SingleSEA


def main():
    # Simulation parameters
    t_start = 0.0      # Start time (s)
    t_end = 100.0      # End time (s)
    dt = 0.001          # Time step (s)
    t = np.arange(t_start, t_end + dt, dt)  # Ensure t_end is included

    # Single SEA parameters
    params = {
        'J_m': 0.44,           # Motor inertia (kg·m²)
        'J_j': 0.29,           # Joint/load inertia (kg·m²)
        'K_s': 1180.0,         # Spring stiffness (N·m/rad)
        'B_m': 17.9,           # Motor damping (N·m·s/rad)
        'link_length': 1.0,    # Length of the link (m)
        'F': 0.0,              # Initial external force magnitude (N)
        'alpha': 0.0,          # Initial external force direction (rad)
        'K1': 0.0,              
        'B1': 0.0,
        'K2': 0.0,
        'B2': 0.0,
        'K_t': 1.0,
    }

    # Initialize SingleSEA system
    single_sea = SingleSEA(params)

    # Initial conditions: [theta_m(0), omega_m(0), theta_s(0), omega_s(0)]
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    # Visualization parameters
    link_length = params['link_length']
    base_position = np.array([0, 0])  # Base position

    # Prepare figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 2)

    # Main animation axis
    ax_anim = fig.add_subplot(gs[0:2, 0])
    ax_anim.set_xlim(-2 * link_length, 2 * link_length)
    ax_anim.set_ylim(-2 * link_length, 2 * link_length)
    ax_anim.set_aspect('equal')
    ax_anim.grid(True)
    ax_anim.set_title('Single SEA Visualization with Interactive Disturbance')

    # State plots axes
    ax_theta_m = fig.add_subplot(gs[0, 1])
    ax_omega_m = fig.add_subplot(gs[1, 1])
    ax_theta_s = fig.add_subplot(gs[2, 0])
    # ax_omega_s = fig.add_subplot(gs[2, 1])
    ax_torque = fig.add_subplot(gs[2, 1])  # Adjust the position as needed


    # Adjust layout for sliders
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, wspace=0.3, hspace=0.7)

    # Initialize plot elements
    line_j, = ax_anim.plot([], [], 'o-', lw=2, label='Joint Link')
    line_m, = ax_anim.plot([], [], 'r-', lw=2, label='Ghost Motor Link')
    disturbance_arrow, = ax_anim.plot([], [], 'g-', lw=2, label='Disturbance Force')
    time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
    ax_anim.legend()

    # Initialize state plots
    theta_m_line, = ax_theta_m.plot([], [], 'b-', label='theta_m')
    theta_j_line, = ax_theta_m.plot([], [], 'c-', label='Joint Position (theta_j)')
    omega_m_line, = ax_omega_m.plot([], [], 'r-', label='omega_m')
    theta_s_line, = ax_theta_s.plot([], [], 'g-', label='theta_s')
    omega_s_line, = ax_omega_m.plot([], [], 'm-', label='omega_s')
    tau_s_line, = ax_torque.plot([], [], 'r-', label='Spring Torque (τ_s)')
    tau_j_line, = ax_torque.plot([], [], 'b-', label='Joint Torque (τ_j)')

    # Set labels and titles for state plots
    ax_theta_m.set_title('Motor and Joint Position (theta_m and theta_j)')
    ax_theta_m.set_xlabel('Time (s)')
    ax_theta_m.set_ylabel('Angle (rad)')
    ax_theta_m.grid(True)

    ax_omega_m.set_title('Motor and Joint Velocity (omega_m and omega_j)')
    ax_omega_m.set_xlabel('Time (s)')
    ax_omega_m.set_ylabel('Angular Velocity (rad/s)')
    ax_omega_m.grid(True)

    ax_theta_s.set_title('Spring Deflection (theta_s)')
    ax_theta_s.set_xlabel('Time (s)')
    ax_theta_s.set_ylabel('Angle (rad)')
    ax_theta_s.grid(True)

    # Set labels and title for torque plot
    ax_torque.set_title('Spring and Joint Torques')
    ax_torque.set_xlabel('Time (s)')
    ax_torque.set_ylabel('Torque (N·m)')
    ax_torque.grid(True)
    ax_torque.legend()

    # Data lists for state plots
    time_data = []
    theta_m_data = []
    theta_j_data = []
    omega_m_data = []
    theta_s_data = []
    omega_s_data = []
    tau_s_data = []
    tau_j_data = []

    # Slider axes
    axcolor = 'lightgoldenrodyellow'
    ax_dist_mag = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
    ax_dist_dir = plt.axes([0.1, 0.12, 0.8, 0.03], facecolor=axcolor)
    ax_k1 = plt.axes([0.1, 0.09, 0.8, 0.03], facecolor=axcolor)
    ax_b1 = plt.axes([0.1, 0.06, 0.8, 0.03], facecolor=axcolor)
    ax_k2 = plt.axes([0.1, 0.03, 0.8, 0.03], facecolor=axcolor)
    ax_b2 = plt.axes([0.1, 0.0, 0.8, 0.03], facecolor=axcolor)
    ax_pos_des = plt.axes([0.1, 0.18, 0.8, 0.03], facecolor=axcolor)


    # Sliders
    slider_dist_mag = Slider(ax_dist_mag, 'Disturbance Magnitude (N)', 0.0, 100.0, valinit=0.0, valstep=1.0)
    slider_dist_dir = Slider(ax_dist_dir, 'Disturbance Direction (rad)', -np.pi, np.pi, valinit=0.0)
    slider_k1 = Slider(ax_k1, 'K1', 0.0, 1000.0, valinit=694.8, valstep=1.0)
    slider_b1 = Slider(ax_b1, 'B1', 0.0, 100.0, valinit=13.87)
    slider_k2 = Slider(ax_k2, 'K2', -1000.0, 0.0, valinit=-416.9, valstep=1.0)
    slider_b2 = Slider(ax_b2, 'B2', -100.0, 0.0, valinit=-11.83)
    slider_pos_des = Slider(ax_pos_des, 'Position Setpoint', -np.pi, np.pi, valinit=0.1)



    # Initial disturbance parameters
    disturbance_magnitude = slider_dist_mag.val
    disturbance_direction = slider_dist_dir.val
    k1 = slider_k1.val
    b1 = slider_b1.val
    k2 = slider_k2.val
    b2 = slider_b2.val
    pos_des = slider_pos_des.val

    # Update disturbance parameters when sliders change
    def update_disturbance_magnitude(val):
        nonlocal disturbance_magnitude
        disturbance_magnitude = val

    def update_disturbance_direction(val):
        nonlocal disturbance_direction
        disturbance_direction = val
        
    def update_k1(val):
        nonlocal k1
        k1 = val

    def update_b1(val):
        nonlocal b1
        b1 = val

    def update_k2(val):
        nonlocal k2
        k2 = val

    def update_b2(val):
        nonlocal b2
        b2 = val

    def update_pos_des(val):
        nonlocal pos_des
        pos_des = val

    slider_dist_mag.on_changed(update_disturbance_magnitude)
    slider_dist_dir.on_changed(update_disturbance_direction)
    slider_k1.on_changed(update_k1)
    slider_b1.on_changed(update_b1)
    slider_k2.on_changed(update_k2)
    slider_b2.on_changed(update_b2)
    slider_pos_des.on_changed(update_pos_des)

    # Initialize state and time
    state = x0.copy()
    t_prev = t[0]

    # Animation initialization function
    def init():
        line_j.set_data([], [])
        line_m.set_data([], [])
        disturbance_arrow.set_data([], [])
        time_text.set_text('')

        # Reset data of state plot lines
        theta_m_line.set_data([], [])
        theta_j_line.set_data([], [])
        omega_m_line.set_data([], [])
        theta_s_line.set_data([], [])
        omega_s_line.set_data([], [])
        tau_s_line.set_data([], [])
        tau_j_line.set_data([], [])

        # Clear data lists
        time_data.clear()
        theta_m_data.clear()
        theta_j_data.clear()
        omega_m_data.clear()
        theta_s_data.clear()
        omega_s_data.clear()
        tau_s_data.clear()
        tau_j_data.clear()

        return (line_j, line_m, disturbance_arrow, time_text, theta_m_line, omega_m_line, theta_s_line, omega_s_line, tau_s_line, tau_j_line)

    # Animation update function
    def update(frame):
        nonlocal state, t_prev

        t_curr = t[frame]
        dt_step = t_curr - t_prev

        if dt_step <= 0:
            dt_step = dt  # Ensure positive time step

        # Update disturbance parameters
        F = disturbance_magnitude
        alpha = disturbance_direction
        K1_gain = k1
        B1_gain = b1
        K2_gain = k2
        B2_gain = b2
        state_des = [pos_des, 0.0, pos_des, 0.0]

        single_sea.set_external_force(F, alpha)
        single_sea.set_gains(K1_gain, B1_gain, K2_gain, B2_gain)
        single_sea.set_desired_state(state_des)

        dxdt = single_sea.dynamics(t_curr, state)

        state += dxdt * dt_step

        # Update t_prev
        t_prev = t_curr

        # Positions of joints and loads
        theta_m = state[0]       # Motor position
        omega_m = state[1]       # Motor velocity
        theta_j = state[2]       # Spring deflection position
        omega_j = state[3]       # Spring deflection velocity

        theta_s = theta_j - theta_m  # Spring deflection
        omega_s = omega_j - omega_m  # Spring velocity

        # End of the link
        x_j = base_position[0] + link_length * np.cos(theta_j)
        y_j = base_position[1] + link_length * np.sin(theta_j)

        # End of the link
        x_m = base_position[0] + link_length * np.cos(theta_m)
        y_m = base_position[1] + link_length * np.sin(theta_m)

        # Compute spring torque
        tau_s = params['K_s'] * theta_s

        # Compute external torque
        tau_ext = F * link_length * np.sin(alpha - theta_j)

        # Compute joint torque
        tau_j = tau_ext - tau_s

        # Update link lines
        line_j.set_data([base_position[0], x_j], [base_position[1], y_j])
        line_m.set_data([base_position[0], x_m], [base_position[1], y_m])

        # Update disturbance arrow
        arrow_scale = 0.5  # Scale factor for the arrow length
        arrow_x = x_j
        arrow_y = y_j
        arrow_dx = arrow_scale * (F / 100.0) * np.cos(alpha)
        arrow_dy = arrow_scale * (F / 100.0) * np.sin(alpha)
        disturbance_arrow.set_data([arrow_x, arrow_x + arrow_dx], [arrow_y, arrow_y + arrow_dy])

        # Update time text
        time_text.set_text(f'Time = {t[frame]:.2f} s')

        # Update state plots
        time_data.append(t_curr)
        theta_m_data.append(theta_m)
        theta_j_data.append(theta_j)
        omega_m_data.append(omega_m)
        theta_s_data.append(theta_s)
        omega_s_data.append(omega_s)
        tau_s_data.append(-tau_s)
        tau_j_data.append(tau_j)

        # Update theta_m plot
        theta_m_line.set_data(time_data, theta_m_data)
        theta_j_line.set_data(time_data, theta_j_data)
        ax_theta_m.relim()
        ax_theta_m.autoscale_view()

        # Update omega_m plot
        omega_m_line.set_data(time_data, omega_m_data)
        omega_s_line.set_data(time_data, omega_s_data)
        ax_omega_m.relim()
        ax_omega_m.autoscale_view()

        # Update theta_s plot
        theta_s_line.set_data(time_data, theta_s_data)
        ax_theta_s.relim()
        ax_theta_s.autoscale_view()

        # Update torque plot
        tau_s_line.set_data(time_data, tau_s_data)
        tau_j_line.set_data(time_data, tau_j_data)
        ax_torque.relim()
        ax_torque.autoscale_view()

        return (line_j, line_m, disturbance_arrow, time_text, theta_m_line, theta_j_line, omega_m_line, theta_s_line, omega_s_line, tau_s_line, tau_j_line)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=dt, blit=False, repeat=False)

    # Show animation
    plt.show()
def wrap_to_pi(theta):
    """
    Wraps an angle in radians to the range [-pi, pi).

    Parameters:
    - theta: angle in radians (float)

    Returns:
    - wrapped_angle: angle in radians within [-pi, pi)
    """
    wrapped_angle = (theta + math.pi) % (2 * math.pi) - math.pi
    return wrapped_angle

if __name__ == '__main__':
    main()