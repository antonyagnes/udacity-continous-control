# Udacity Deep Reinforcement Learning Nanodegree

# Project 2: Continuous Control 
For this project, I worked with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment and tried to solve Version 2 (contains 20 identical agents, each with its own copy of the environment). Each agent is a double-jointed arm that can move to target locations.
I solved this problem using  [DDPG](https://arxiv.org/abs/1509.02971) algorithm. 
### DPPG algorithm
##### Basic Actor-Critic Method
Actor-Critic agent is an agent that uses function approximation to learn the policy and the value function.
Actor and critic are two neural networks that function as follows:
**Actor** network is used as an approximate for the optimal deterministic policy. It takes the state as input and outputs distribution over the actions.
**Critic** will learn to evaluate the state value function using the TD estimate. Using this,  advantage function is be calculated. Critic takes state as its input and outputs a state value function.
#### DDPG
DDPG is a model-free kind of actor-critic (basic actor-critic method is defined earlier in this report) method. It uses a different king of actor-critic agent and it can also be called as an approximate DQN since the critic in DDPG is used to maximize over the Q values for the next state and is not used as a learned baseline.
DDPG 4 neural networks: local actor,  target actor, local critic and target critic. The important feature of this algorithm is that it uses Replay Buffers (past experiences are saved so the agent can fetch random samples  from the buffer and learn from it) and performs soft updates (updates the weights of the target networks).
### Implementation
#### Hyperparameters used:
BUFFER_SIZE = int(1e6)  # replay buffer size

BATCH_SIZE = 1024       # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor

LR_CRITIC = 3e-4        # learning rate of the critic

WEIGHT_DECAY = 0.0   # L2 weight decay

#### Actor Network
Actor (policy) network (three fully connected layered network) maps states to actions. The first layer get the state and passes it through a hidden layer with 256 nodes(uses relu as it activation function). The output of the first layer is passed into the second hidden layer with 128 nodes (uses relu as it activation function). Output of the second layer is passed into the third layer (output layer with 4 (action_size) nodes). Uses Adams optimizer. Batch normalization is applied to the state input and before all layers.
#### Critic network
Critic (value) network (three fully connected layered network) that maps (state, action) pairs to Q-values.  The first layer get the state and passes it through a hidden layer with 400 nodes(uses relu as it activation function). The output of the first layer is passed into the second hidden layer with 300 nodes (uses relu as it activation function). Output of the second layer is passed into the third layer (output layer with 1 node). Uses Adams optimizer.  Batch normalization is applied to the state input.
#### Batch Normalization
When learning from low dimensional feature vector observations, the different components of the observation may have different physical units (for example, positions versus velocities) and the ranges may vary across environments. This can make it difficult for the network to learn effectively and may make it difficult to find hyper-parameters which generalize across environments with different scales of state values. This issue is  addressed by adding batch normalization. This technique normalizes each dimension across the samples in a minibatch to have unit mean and variance. In addition, it maintains a running average of the mean and variance to use for normalization during testing (in our case, during exploration or evaluation).
In our case,  batch normalization to the state input and before all layers in the actor network. The critic network also uses batch normalization on the state inputs, action signals will not be altered.
#### Noise
A major challenge of learning in continuous action spaces is exploration. To address this issue noise class is added (Ornstein-Uhlenbeck process).
### Reward Plot
The agent was able to reach a score of 30.0 at 210th episode and it was able to maintain score > 30.0 until 400th episode.
![enter image description here](https://lh3.googleusercontent.com/L7yEWqOojU9WwkcNstsHj6VnYz3ZHZb0Om6zdkvGAFB1pTPgLpRLvHjIejqB2qrjwA7DN-ZfWmBzjiua5mej2IV3SpfHVbQy68TANquFpdEvpFYFmEXcl2gA5GPhWX0y2LDTU712YPKeu_yZkvFpfOZ1dAA4X1pSTvYOxLbLGSo9nLnVGjoIkljCsRN9gx-d1ODmuLd6y2YPX19_N6wpU350xmpKGh_TKZKFGugDEUNT-aONVFtLMBxAFpX-B7CS1eUswQ28ttAutC5ZmOM6UAgHYK9rKpwysHnIu6kl0AZQnmE5WMQQBtww5RiEFLnVHTEzADpYyFaoLjTbbaKfS2QcSkERqY_ITZ_glX4EzKU9w3reZhiPybG-6AB5BfwHtO5RlKiIullNqyvIczfggTZuHycb0fB_dbg8FGZhkj6nKoKWBf8AqcugeqOY0XaGb1WfNK8M6BNqxW8KBWEOkBCrBdwrtsD-sTHAr78aVspallN1H0ga0T6CYA2wRIO0MdKPKjvOmqvYJSHTxpnsRuTW3PyQIyqfqnpdrggibtM9eIYpJResmzwnciJ_7IaEmSBXRLYiF_HD8qCM2T-B0O223qipTvvdATGqG591xPrfIU4cJU4TbZQ_gtjdVhy1IINIJoOh7KJd75wSI02yQFhsQXTdjTsdNFBuaFHCyl6Hrcu2NHTs9FsQuaSTPRcOL4CaC29FY-3IYuY9WWW5jHkL=w382-h262-no?authuser=0)
### Ideas for Future Work
- Try different hyper-parameters (luckily the first set of hyper-parameters thats were used seemed to yield the expected result).
- Use different algorithms like PPO and compare the results.
