# 🌟 **Machine Learning Project: Comparing RL Models on a Maze** 🌟

## 🚀 **Introduction**

Welcome to our **Reinforcement Learning Maze Project**! 🎯 This project compares the performance of two popular RL algorithms, **SARSA** and **Monte Carlo**, as they learn to navigate a maze. 🗺️ Our goal is to evaluate their efficiency, learning strategies, and overall behavior in solving the maze puzzle. 🧩  

By the end of this project, you'll gain insights into the strengths and weaknesses of these algorithms while observing their performance visually! 📊✨

---

## 🎓 **Course and Contributors**

This project is part of the **Masters in Autonomous Systems** program for the **Machine Learning** course, under the mentorship of **Professor Sebastian Houben**.  

### 🤝 **Contributors**
- **Prachi Sheth** 🌟  
- **Amol Tatkari** 🌟  
- **Vedika Chauhan** 🌟  
- **Trushar Ghanekar** 🌟  

---

## 📚 **Reinforcement Learning Models**

In this project, we evaluate two reinforcement learning algorithms:

### 1️⃣ **SARSA**  
An **on-policy** algorithm that learns the value of the policy being executed. It updates Q-values at each step of an episode using the agent's current policy. ⚙️

### 2️⃣ **Monte Carlo**  
A **model-free** algorithm that relies on episodic sampling. It computes Q-values based on the cumulative returns observed at the end of an episode. 🔄

---

## 🧠 **How They Work**

### 🟦 **SARSA Algorithm**  
SARSA stands for **State-Action-Reward-State-Action** and follows these steps:
1. **State (S)**: Start in a state in the maze.  
2. **Action (A)**: Choose an action based on the policy (e.g., epsilon-greedy).  
3. **Reward (R)**: Receive feedback based on the action taken.  
4. **Next State (S')**: Move to a new state.  
5. **Next Action (A')**: Choose the next action based on the policy.  

#### Formula for Updating Q-values:
$[
Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q(S', A') - Q(S, A) \right]
]$

- **Parameters**:  
- $( \alpha )$: Learning rate.
- $( \gamma )$: Discount factor, which determines the importance of future rewards.
- $( R )$: Reward for taking action \( A \) in state \( S \).
- $( S' )$, $( A' )$: The next state and action.

---

### 🟩 **Monte Carlo Algorithm**  
Monte Carlo learns by sampling **complete episodes** and updating Q-values based on cumulative returns. 🌐

#### Steps:
1. **Initialize Q-Table**: Start with an empty or randomly initialized table.  
2. **Episode Generation**: Interact with the maze environment to generate episodes.  
3. **Calculate Return (G)**:
   - Compute the **return** $( G )$ for each state-action pair in the episode:
     $(
     G = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots
     )$
     where $( \gamma )$ is the discount factor.
4. **Update Q-values**:
   - For each state-action pair in the episode, update the Q-value as the **average of observed returns**.  

---

## 🔍 **Environment and Visualization**

The environment used is the **FrozenLake** maze from OpenAI Gym 🧊.  
- The maze is represented as a gridworld where the agent learns to navigate to the goal efficiently. 🏁  

**Visualization Tools**:  
- `Visualization_Monte_Carlo.py` 🟩: For visualizing the Monte Carlo algorithm.  
- `sarsa_visualization.py` 🟦: For visualizing the SARSA algorithm.  

---

## 🏆 **Comparison and Observations**

| 🔢 **Algorithm**   | 🏅 **Average Reward** | ⏱️ **Training Time (s)** | 📜 **Policy Type** |
|--------------------|-----------------------|--------------------------|--------------------|
| **SARSA**          | TBD                   | 0.05                    | On-policy          |
| **Monte Carlo**    | TBD                   | 0.22                    | Off-policy         |

### 🗝️ **Key Observations**
- **SARSA**: Learns from each step during training, adapting based on its policy. It is sensitive to the current policy being executed.  
- **Monte Carlo**: Relies on complete episodes, making it slower but more precise in environments with well-defined episodic tasks.  

---

## 📂 **Project Files**

1. **`ML_Project_Both_Algorithm.ipynb`**: Contains the implementation and comparison of SARSA and Monte Carlo algorithms.  
2. **`Visualization_Monte_Carlo.py`**: Script for visualizing the Monte Carlo algorithm.  
3. **`sarsa_visualization.py`**: Script for visualizing the SARSA algorithm.  

---

## 📖 **References**

- 🌐 [SARSA Reinforcement Learning](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)  
- 🌐 [Reinforcement Learning Maze Project](https://github.com/erikdelange/Reinforcement-Learning-Maze)  

---

💡 **Happy Learning!** 😊
