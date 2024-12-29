# Machine Learning Project: Comparing RL Models on a Maze

## Introduction

This project compares two reinforcement learning (RL) models in the context of navigating a maze. The aim is to evaluate the performance and behavior of these models as they learn to find the most efficient path through the maze. By comparing these models, the project highlights their strengths, weaknesses, and learning strategies.

## Reinforcement Learning Models

Two RL models are implemented and evaluated:

1. **SARSA**
2. **Monte Carlo**

The comparison involves measuring their success in solving the maze, and the time taken to run the episodes.

---

## SARSA Algorithm

The SARSA (State-Action-Reward-State-Action) algorithm is an on-policy RL algorithm, meaning it learns the value of the policy that is being executed, including the actions taken by the agent during training.

### How SARSA Works

1. **State (S)**: The agent starts in a state in the environment (the maze).
2. **Action (A)**: The agent selects an action based on its policy (e.g., epsilon-greedy), which dictates how likely the agent is to explore new actions versus exploiting the best-known action.
3. **Reward (R)**: The agent receives a reward (or penalty) based on the action taken and the resulting state.
4. **Next State (S')**: The agent transitions to a new state based on its action.
5. **Next Action (A')**: The agent selects the next action according to its policy in the new state.

SARSA updates the Q-value (the expected future reward for a state-action pair) using the following formula:

\[
Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]
\]

Where:
- \( \alpha \): Learning rate.
- \( \gamma \): Discount factor, which determines the importance of future rewards.
- \( R \): Reward for taking action \( A \) in state \( S \).
- \( S' \), \( A' \): The next state and action.

---

## Monte Carlo Algorithm

The Monte Carlo (MC) algorithm is a model-free reinforcement learning technique used to estimate optimal policies for Markov Decision Processes (MDPs). It relies on the concept of episodic sampling, where an agent interacts with the environment, collects data in the form of episodes, and updates its Q-values based on the returns (cumulative rewards) observed from these episodes.

### How Monte Carlo Works

1. **Initialization**: Start with an empty or randomly initialized Q-table.
2. **Episode Generation**: Generate episodes by interacting with the environment using an epsilon-greedy policy.
3. **Return Calculation**:
   - Compute the **return** \( G \) for each state-action pair in the episode:
     \[
     G = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots
     \]
     where \( \gamma \) is the discount factor.
4. **Update Q-Values**:
   - For each state-action pair encountered in the episode:
     - Append the return \( G \) to the list of returns for that pair.
     - Update the Q-value as the average of the accumulated returns.

---

## Comparison and Observations

### Visualization
The environment used for this project is a maze visualized using the FrozenLake environment from OpenAI Gym. The maze represents a gridworld where the agent must learn to navigate to the goal efficiently.

### Key Observations
- **SARSA**: Learns from each step in an episode, adapting based on the policy being executed. It is sensitive to the current policy.
- **Monte Carlo**: Learns from complete episodes, making it slower but potentially more accurate for episodic environments.

### Performance Table

| Algorithm    | Average Reward | Training Time (s) | Policy Type |
|--------------|----------------|--------------------|-------------|
| **SARSA**    | TBD            | 0.05                | On-policy   |
| **Monte Carlo** | TBD          | 0.22                | Off-policy  |

---

## References

- [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)
- [Reinforcement Learning Maze Project](https://github.com/erikdelange/Reinforcement-Learning-Maze)
