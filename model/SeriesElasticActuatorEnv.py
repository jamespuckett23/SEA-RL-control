from typing import Optional
import numpy as np
import gymnasium as gym

class SeriesElasticActuatorEnv(gym.Env):
    def __init__(self, inputs):
        self.motor_theta = np.float32
        self.motor_theta_dot = np.float32
        self.spring_theta = np.float32
        self.spring_theta_dot = np.float32

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
                "motor_theta": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
                "motor_theta_dot": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
                "spring_theta": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
                "spring_theta_dot": gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32),
            }
        )

    def _get_obs(self):
        return {"motor_theta": self.motor_theta, "motor_theta_dot": self.motor_theta_dot, "spring_theta": self.spring_theta, "spring_theta_dot": self.spring_theta_dot}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[np.float32] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the motor's spring to be at any position between -pi and pi
        self.motor_theta = seed
        self.motor_theta_dot = 0.0

        # Assume simple case where the spring offset is reset to be zero
        self.spring_theta = 0.0
        self.spring_theta_dot = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: np.float32):
        success = self.send_current_cmd(action)

        if not success:
            print("Did not send command successfully")
        
        self.motor_theta = self.read_motor_sensor()
        self.spring_theta = self.read_spring_sensor()

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def read_motor_sensor(self):
        return np.float32
    
    def read_spring_sensor(self):
        return np.float32
    
    def send_current_cmd(self, current_cmd: np.float32):
        success = True
        return success