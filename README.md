# An RL-Based Full-State Feedback Controller for Series Elastic Actuators

## 📜 Description

This project approaches the full state feedback control problem for series elastic actuators from a reinforcement learning perspective. Traditional full state feedback control techniques do no address nonlinearities such as spring hysteresis, gear backlash, and friction. With RL there is potential to learn how to compensate for such nonlinearities. This project includes intends to create a simulation environment for training control algorithms for series elastic actuators and a hardware interface to control series elastic actuators via the ROS2 software framework. 

## 🛠️ Features

- Dynamic Simulation of a Single Degree of Freedom Series Elastic Actuator
- An OpenAI Gym Environment to train RL algorithms on how to control it
- Scripts to run the trained model both in simulation and on hardware using the ROS2 robotics software framework

### Prerequisites
This project uses the same OpenAI gym set up that the CSCE 642 assignments used here: https://people.engr.tamu.edu/guni/csce642/assignments.html. For the hardware testing it requires ROS2 humble installed on your computer and HERC lab proprietary software and hardware. HERC lab software and hardware are not included in this repository.  

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/jamespuckett23/SEA-RL-control.git
2. Run the AI trained model in simulation:
   ```bash
   cd model
   python visualize_single_sea_rl.py
3. Run the FSF Feedback Controller in simulation:
   ```bash
   cd model
   python visualize_single_sea.py
