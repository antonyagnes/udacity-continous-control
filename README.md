
# Udacity Deep Reinforcement Learning Nanodegree

# Project 2: Continuous Control 
![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)
## Goal
In this project, a double-jointed arm (20 agents in this case) should be able to move to target locations. Thus, the goal of the agent is to maintain its position at target location for as many time steps as possible.

## Environment
The environment provided by [Udacity](www.udacity.com) is similar to the one built by [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) .

### State space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

### Action space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Reward
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an  **average score**  for each episode (where the average is over all 20 agents).

### Expected result
The environment is considered solved, when the average (over 100 episodes) of those **average scores** is at least +30. 

## Set up Instructions
Please follow the  [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)  to set up your Python environment. These instructions can be found in  `README.md`  at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Getting Started
For this project, you will **not** need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Download the environment from one of the links below. You need only select the environment that matches your operating system:
    ### Version 2: Twenty (20) Agents

-   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
-   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
-   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
-   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the  `p2_continuous-control/`  folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(_For Windows users_) Check out  [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)  if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (_To watch the agent, you should follow the instructions to  [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the  **Linux**  operating system above._)
    
 **Note:**  While setting it up on Windows, I also had to install mlagents `pip install mlagents==0.4.0` 

### Included in this repository
-   The code used to create and train the Agent
    -   Continuous_Control.ipynb
    -   dppg_agent.py
    -   model.py 
-   The trained model
    -   checkpoint_actor.pth
    -   checkpoint_critic.pth
-   A Report.pdf file describing the development process and the learning algorithm, along with ideas for future work

### Instructions to run the code
Open  `Continuous_Control.ipynb` and follow the instructions in the notebook.

