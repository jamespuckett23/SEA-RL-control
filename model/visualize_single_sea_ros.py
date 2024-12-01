import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from DDPG import DDPG
import torch
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SEAControlNode(Node):
    def __init__(self):
        super().__init__('sea_control_node')
        
        # ROS 2 Subscribers and Publishers
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10
        )
        self.publisher = self.create_publisher(Float32, '/desired_torque', 10)

        # Initialize parameters
        self.motor_position = 0.0
        self.motor_velocity = 0.0
        self.spring_position = 0.0
        self.spring_velocity = 0.0
        self.spring_stiffness = 200.0  # Nm/rad

        # Load trained AI model
        class Options:
            def __init__(self):
                self.gamma = 0.99
                self.alpha = 0.001
                self.epsilon = 0.5
                self.epsilon_min = 0.01
                self.epsilon_decay = 0.999
                self.num_episodes = 5000
                self.steps = 1000
                self.layers = [128, 128, 64]
                self.replay_memory_size = 500000
                self.batch_size = 64
                self.update_target_estimator_every = 500

        options = Options()
        self.actor_model = DDPG(None, None, options)
        self.actor_model.actor_critic.load_state_dict(torch.load("actor_critic_5000_from_good_version.pth"))
        self.actor_model.target_actor_critic.load_state_dict(torch.load("target_actor_5000_from_good_version.pth"))

        # Initial desired state and action
        self.desired_position = 0.0
        self.desired_torque = 0.0

        # Setup the visualization
        self.setup_visualization()

    def joint_states_callback(self, msg):
        # Extract motor position and velocity from the joint states message
        self.motor_position = msg.position[0]  # Assuming the motor is the first joint
        self.motor_velocity = msg.velocity[0]

        # Calculate spring position from the effort (assuming the spring is the second joint)
        spring_effort = msg.effort[0]
        self.spring_position = spring_effort / self.spring_stiffness

        # Set spring velocity to zero for simplicity (if not available)
        self.spring_velocity = 0.0

        # Update the state for the agent
        state = np.array([self.motor_position, self.motor_velocity, self.spring_position, self.spring_velocity, self.desired_position, self.desired_torque])
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # Get action from the agent
        torque_action = self.actor_model.select_action(state_tensor)
        
        # Publish the torque command
        msg = Float32()
        msg.data = torque_action[0]
        self.publisher.publish(msg)

        # Update the visualization
        self.update_visualization()

    def setup_visualization(self):
        # Setup visualization with matplotlib
        self.fig, self.ax = plt.subplots()
        self.link_length = 1.0
        self.base_position = np.array([0, 0])

        self.ax.set_xlim(-2 * self.link_length, 2 * self.link_length)
        self.ax.set_ylim(-2 * self.link_length, 2 * self.link_length)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('Single SEA Visualization with AI Controller')

        self.line_j, = self.ax.plot([], [], 'o-', lw=2, label='Joint Link')
        self.line_m, = self.ax.plot([], [], 'r-', lw=2, label='Motor Link')
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.ax.legend()

        # Slider setup
        axcolor = 'lightgoldenrodyellow'
        self.ax_pos_des = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
        self.slider_pos_des = Slider(self.ax_pos_des, 'Position Setpoint', -np.pi, np.pi, valinit=0.0)
        self.slider_pos_des.on_changed(self.update_desired_position)

        self.ani = animation.FuncAnimation(self.fig, self.update_visualization, interval=50, blit=False, repeat=False)
        plt.show()

    def update_desired_position(self, val):
        self.desired_position = val

    def update_visualization(self, *args):
        # Visualize the links based on the motor and spring positions
        theta_m = self.motor_position
        theta_j = self.spring_position + self.motor_position

        x_j = self.base_position[0] + self.link_length * np.cos(theta_j)
        y_j = self.base_position[1] + self.link_length * np.sin(theta_j)
        x_m = self.base_position[0] + self.link_length * np.cos(theta_m)
        y_m = self.base_position[1] + self.link_length * np.sin(theta_m)

        self.line_j.set_data([self.base_position[0], x_j], [self.base_position[1], y_j])
        self.line_m.set_data([self.base_position[0], x_m], [self.base_position[1], y_m])

        # Update the time text
        self.time_text.set_text(f'Time = {self.get_clock().now().to_msg().sec:.2f}s')
        
        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = SEAControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
