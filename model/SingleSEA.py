# single_sea_simulation.py

import numpy as np

class SingleSEA:
    def __init__(self, params):
        # System parameters
        self.J_m = params['J_m']
        self.J_j = params['J_j']
        self.K_s = params['K_s']
        self.B_m = params['B_m']
        self.link_length = params['link_length']
        self.K_t = params['K_t']

        # External force parameters
        self.F = params.get('F', 0.0)         # Force magnitude
        self.alpha = params.get('alpha', 0.0) # Force direction (rad)

        self.K1 = params.get('K1', 0.0)
        self.B1 = params.get('B1', 0.0)
        self.K2 = params.get('K2', 0.0)
        self.B2 = params.get('B2', 0.0)

        self.x_des = np.array([0.0, 0.0, 0.0, 0.0])
        self.A = np.array([[               0.0,                1.0,                0.0, 0.0],
                           [-self.K_s/self.J_m, -self.B_m/self.J_m,  self.K_s/self.J_m, 0.0],
                           [               0.0,                0.0,                0.0, 1.0],
                           [ self.K_s/self.J_j,                0.0, -self.K_s/self.J_j, 0.0]])
        self.B = np.array([[         0.0,          0.0],
                           [1.0/self.J_m,          0.0], 
                           [         0.0,          0.0], 
                           [         0.0, 1.0/self.J_j]]) 
        
        self.max_torque = 350.0
        self.ff_tau = 0.0
        self.use_gains = True

    def dynamics(self, t, x):
        """
        Computes the derivatives for the Single SEA system.

        Parameters:
        - t: Time (scalar)
        - x: State vector [theta_m, omega_m, theta_s, omega_s]

        Returns:
        - dxdt: Derivative of state vector
        """

        tau_m = self.fsf_controller(x, self.x_des)

        # External torque due to force F applied at angle alpha
        theta_j = x[2]
        tau_ext = self.F * self.link_length * np.sin(self.alpha - theta_j)

        u = np.array([tau_m, tau_ext])

        dxdt = np.dot(self.A, x.T) + np.dot(self.B, u.T)
        return dxdt

    def set_external_force(self, F, alpha):
        # Update external force parameters
        self.F = F
        self.alpha = alpha
    
    def set_gains(self, K1, B1, K2, B2):
        self.K1 = K1
        self.B1 = B1
        self.K2 = K2
        self.B2 = B2
    
    def set_desired_state(self, state_desired):
        self.x_des = state_desired
    
    def set_K_t(self, K_t):
        self.K_t = K_t

    def set_ff_tau(self, current):
        self.ff_tau = self.K_t * current

    def set_use_gains(self, use_gains):
        self.use_gains = use_gains

    def fsf_controller(self, x, x_des):
        m_tau = self.ff_tau
        if self.use_gains:
            # Unpack state variables
            theta_m, omega_m, theta_s, omega_s = x
            theta_m_des, omega_m_des, theta_s_des, omega_s_des = x_des
            theta_m_err = self.K1 * (x_des[0] - x[0])
            omega_m_err = self.B1 * (x_des[1] - x[1])
            theta_s_err = self.K2 * ((x_des[2] - x_des[0]) - (x[2] - x[0]))
            omega_s_err = self.B2 * ((x_des[3] - x_des[1]) - (x[3] - x[1]))
            m_tau += theta_m_err + omega_m_err + theta_s_err + omega_s_err
        return self.clamp(m_tau, -self.max_torque, self.max_torque)
    
    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))



