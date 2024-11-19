Notes:

Currently uses spares rewards and long horizon
- Reward should shift to be gradient of each term -> when you move in the currect direction, get a positive reward. 
- Possibly add physics informed-neural network or neural network for rl model


Decay learning rate over the episodes
Only control the torque, and then fill in the state space -> is the state space adjust appropriately to the actions? I believe it is
Scale rewards
Correct positional reward (~-pi ~= ~pi)
Add positive rewards during episode? less negative is about the same as positive


Write testing script