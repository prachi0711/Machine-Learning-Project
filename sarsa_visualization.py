import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def initialize_q_table(states, actions):
    """Initialize the Q-table with small random values."""
    return np.zeros((states, actions))

def choose_action(state, q_table, epsilon):
    """Choose an action using epsilon-greedy strategy."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """Update the Q-value using the SARSA update rule."""
    state, action, next_state, next_action = int(state), int(action), int(next_state), int(next_action)
    td_target = reward + gamma * q_table[next_state, next_action]
    td_error = td_target - q_table[state, action]
    q_table[state, action] += alpha * td_error

def sarsa(env, episodes, alpha, gamma, epsilon, epsilon_decay):
    """SARSA algorithm implementation."""
    q_table = initialize_q_table(env.observation_space.n, env.action_space.n)
    start_time = time.time()
    rewards=[]

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        action = choose_action(state, q_table, epsilon)
        total_reward = 0
        
        print(f"Episode {episode + 1}/{episodes} (SARSA)")

        while True:
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, done, truncated, _ = result
            elif len(result) == 4:
                next_state, reward, done, truncatesd = result
            else:
                raise ValueError("Unexpected return format from env.step()")

            if isinstance(next_state, tuple):
                next_state = next_state[0]
            
            next_action = choose_action(next_state, q_table, epsilon)
            update_q_table(q_table, state, action, reward, next_state, next_action, alpha, gamma)

            total_reward += reward
            state, action = next_state, next_action
            
            if done or truncated:
                rewards.append(total_reward)
                print(f"Episode finished with total reward: {total_reward}\n")
                break

        epsilon = max(epsilon * epsilon_decay, 0.01)

    elapsed_time = time.time() - start_time
    print(f"Total time for SARSA: {elapsed_time:.2f} seconds")
    return q_table,rewards

def visualize_sarsa_policy(env, q_table):
    """Visualize the policy derived from the SARSA Q-table."""
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    env.render()
    total_reward = 0

    print("\nVisualizing SARSA Policy:")
    while True:
        action = np.argmax(q_table[state])  # Choose the best action from Q-table
        result = env.step(action)

        if len(result) == 5:
            next_state, reward, done, truncated, _ = result
        elif len(result) == 4:
            next_state, reward, done, truncated = result
        else:
            raise ValueError("Unexpected return format from env.step()")

        if isinstance(next_state, tuple):
            next_state = next_state[0]

        total_reward += reward
        state = next_state

        env.render()  # Display the current state of the environment
        time.sleep(0.5)  # Pause for better visualization

        if done or truncated:
            print(f"Finished with total reward: {total_reward}")
            break
def plot_state_values(maze, q_table, rewards,episodes, grid_size, title="State Values"):
    """
    Visualize the state action values Q in each walkable square after training.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    policy=[]
    for i in range(q_table.shape[0]):
        policy.append(q_table[i][np.argmax(q_table[i])])

    policy=np.array(policy)
    policy=np.reshape(policy,(4,4))

    # Draw the maze grid
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            cell_value = maze[row, col]

            # Walls or Holes
            if cell_value == b'H':  # 'H' is for holes in FrozenLake
                ax.add_patch(plt.Rectangle((col, grid_size[0] - row - 1), 1, 1, color="gray"))
                ax.text(col + 0.5, grid_size[0] - row - 0.5, "-", ha="center", va="center", fontsize=16, color="black")
            # Start or Goal
            elif cell_value in [b'S', b'G']:
                ax.add_patch(plt.Rectangle((col, grid_size[0] - row - 1), 1, 1, color="lightblue"))
                ax.text(col + 0.5, grid_size[0] - row - 0.5, cell_value.decode("utf-8"), ha="center", va="center", fontsize=16, color="black")
            # Walkable cells
            elif cell_value == b'F':  # 'F' is for frozen walkable cells
                state = row * grid_size[1] + col
                state_value = policy[row,col]  # Default to 0.00 if state is missing
                ax.text(col + 0.5, grid_size[0] - row - 0.5, f"{state_value:.3f}", ha="center", va="center", fontsize=12, color="blue")

    # Draw grid lines
    for i in range(grid_size[0] + 1):
        ax.plot([0, grid_size[1]], [i, i], color='black', linewidth=1)
    for j in range(grid_size[1] + 1):
        ax.plot([j, j], [0, grid_size[0]], color='black', linewidth=1)

    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.show()

    #Plot Reward Data
    g=plt.figure(2)
    plt.plot(episodes,rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    g.show()
    plt.show()


if __name__ == "__main__":
    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

    # Hyperparameters
    episodes = 5000
    alpha = 0.5  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.999

    # Train SARSA
    q_table,rewards = sarsa(env, episodes, alpha, gamma, epsilon, epsilon_decay)

    # Visualize the learned policy
    print("\n--- Visualizing SARSA Policy ---")
    visualize_sarsa_policy(env, q_table)

    maze = np.array(env.desc)
    episodes=np.linspace(1,episodes,episodes)

    #PLotting Results
    plot_state_values(maze, q_table, rewards, episodes, grid_size=(maze.shape[0], maze.shape[1]), title="Q Values After Training")
    
