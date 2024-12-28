# Machine Learning Project: Comparing RL Models on a Maze

## Introduction

This project compares two reinforcement learning (RL) models in the context of navigating a maze. The aim is to evaluate the performance and behavior of these models as they learn to find the most efficient path through the maze. By comparing these models, the project highlights their strengths, weaknesses, and learning strategies.

## Reinforcement Learning Models

Two RL models are implemented and evaluated:

1. **Model 1**: SARSA
2. **Model 2**: 

The comparison involves measuring their success in solving the maze, and the time taken to run the episodes.

## SARSA Algorithm

The SARSA (State-Action-Reward-State-Action) algorithm is one of the reinforcement learning algorithms used in this project. It is an on-policy algorithm, meaning it learns the value of the policy that is being executed, including the actions taken by the agent during training.

### How SARSA Works:

1. **State (S)**: The agent starts in a state in the environment (the maze).
2. **Action (A)**: The agent selects an action based on its policy (e.g., epsilon-greedy), which dictates how likely the agent is to explore new actions versus exploiting the best-known action.
3. **Reward (R)**: The agent receives a reward (or penalty) based on the action taken and the resulting state.
4. **Next State (S')**: The agent transitions to a new state based on its action.
5. **Next Action (A')**: The agent selects the next action according to its policy in the new state.

SARSA updates the Q-value (the expected future reward for a state-action pair) using the following formula:

$`
Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]
`$

Where:
- $` \alpha `$ is the learning rate.
- $` \gamma `$ is the discount factor, which determines the importance of future rewards.
- R is the reward for taking action A in state S.
- S' and A' are the next state and action.

### XYZ Algorithm:
the description of other algorithm used

### Visualization:
environment that is being used for visualization

### Observation:
Comparision of the two algorithms by creating a table

### References:
- ![SARSA Model](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)
- ![RL Models for maze](https://github.com/erikdelange/Reinforcement-Learning-Maze)
