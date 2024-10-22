from typing import Optional
from SingleSEA import SingleSEA
import numpy as np
import gymnasium as gym

class SeriesElasticActuatorEnv(gym.Env):
    def __init__(self, inputs):
        self.motor_theta = np.float32
        self.motor_theta_dot = np.float32
        self.joint_theta = np.float32
        self.joint_theta_dot = np.float32

        # include goals to input
        self.goal_theta = 1.0
        self.goal_theta_dot = 1.0

        # Initialize the SingleSEA system with default parameters
        params = {
            'J_m': 0.44,           # Motor inertia (kg·m²)
            'J_j': 0.29,           # Joint/load inertia (kg·m²)
            'K_s': 1180.0,         # Spring stiffness (N·m/rad)
            'B_m': 17.9,           # Motor damping (N·m·s/rad)
            'link_length': 1.0,    # Length of the link (m)
            'F': 0.0,              # Initial external force magnitude (N)
            'alpha': 0.0,          # Initial external force direction (rad)
            'K_t': 100.0,           # Current to motor torque constant
        }

        self.system = SingleSEA(params)

        # Required to initialize the action space and observation space for the gym environment

        # The action space is a current command between -1A and 1A
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 0), dtype=np.float32)

        # The observation space contains the information about the spring and motor
        # At the moment, it also provides simple processing for 
        #   the respective velocities of the spring and motor
        # Currently, the data range is from -100 to 100 which should cover any possible range
        # Theta should be mapped from -2pi to 2pi, -pi to pi (probably the best), or 0 to 2pi
        self.observation_space = gym.spaces.Dict(
            {
                "motor_theta": gym.spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "motor_theta_dot": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
                "joint_theta": gym.spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "joint_theta_dot": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
            }
        )

    def _get_obs(self):
        return {"motor_theta": self.motor_theta, "motor_theta_dot": self.motor_theta_dot, "joint_theta": self.joint_theta, "joint_theta_dot": self.joint_theta_dot}

    def _get_info(self):
        return {
            "distance to goal": np.linalg.norm(
                self.motor_theta - self.goal_theta, ord=2
            )
        }

    def reset(self, seed: Optional[np.float32] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the motor's spring to be at any position between -pi and pi
        self.motor_theta = seed
        self.motor_theta_dot = 0.0

        # Assume simple case where the spring offset is reset to be zero
        self.joint_theta = self.motor_theta
        self.joint_theta_dot = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.float32):
        success = self.send_current_cmd(action)

        if not success:
            print("Did not send command successfully")
        
        # update state from sensors
        self.motor_theta = self.read_motor_sensor()
        self.joint_theta = self.read_joint_sensor()
        self.motor_theta_dot = self.read_motor_sensor()
        self.joint_theta_dot = self.read_joint_sensor()

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self.motor_theta, self.goal_theta) and np.array_equal(self.motor_theta_dot, self.goal_theta_dot)
        truncated = False
        reward = self.get_reward(terminated)
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def read_motor_sensor(self):
        return np.float32
    
    def read_joint_sensor(self):
        return np.float32
    
    def send_current_cmd(self, current_cmd: np.float32):
        success = True
        return success
    
    def get_reward(self, terminated):
        position_control_weight = 1.0
        velcoity_control_weight = 1.0
        response_weight = 1.0

        if terminated:
            reward = 100.0 # adjust to find an accurate goal achievement
        else:
            reward = position_control_weight * abs(self.motor_theta - self.joint_theta) + \
                     velcoity_control_weight * abs(self.motor_theta_dot - self.joint_theta_dot) + \
                     response_weight * (self.system.K_s * abs(self.goal_theta - self.joint_theta) + \
                                        self.system.B_m * abs(self.goal_theta_dot - self.joint_theta_dot) - \
                                        self.system.k_s * abs(self.motor_theta))
        
        return reward