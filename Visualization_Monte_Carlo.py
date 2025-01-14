import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt

from collections import defaultdict


def choose_action(state, q_table, epsilon, action_space):
    """Choose an action using epsilon-greedy strategy."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, action_space - 1)
    else:
        return np.argmax(q_table[state])


def monte_carlo(env, episodes, gamma, epsilon, epsilon_decay):
    """Monte Carlo control with epsilon-greedy policy."""
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)  # Store returns for state-action pairs
    start_time = time.time()
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_data = []  # To store state, action, reward
        total_reward = 0

        print(f"Monte Carlo Episode {episode + 1}/{episodes}")

        while True:
            action = choose_action(state, q_table, epsilon, env.action_space.n)
            result = env.step(action)
            next_state, reward, done, truncated, _ = result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state

            # Render every 10 episodes for visualization
            #if episode % 10 == 0:
                #env.render()

            if done or truncated:
                print(f"Episode finished with total reward: {total_reward}\n")
                break
        
        episode_rewards.append(total_reward)

        # Calculate returns and update Q-values
        G = 0  # Initialize return
        visited = set()

        for state, action, reward in reversed(episode_data):
            G = reward + gamma * G
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                q_table[state][action] = np.mean(returns[(state, action)])

        epsilon = max(epsilon * epsilon_decay, 0.01)

    elapsed_time = time.time() - start_time
    print(f"Total time for Monte Carlo: {elapsed_time:.2f} seconds")
    return q_table, elapsed_time, episode_rewards


def visualize_policy(env, q_table):
    """Visualize the policy derived from the Q-table."""
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    env.render()
    total_reward = 0

    print("\nVisualizing Optimal Policy:")
    while True:
        action = np.argmax(q_table[state])
        result = env.step(action)
        state, reward, done, truncated, _ = result
        total_reward += reward
        env.render()
        time.sleep(0.5)  # Pause for better visualization

        if done or truncated:
            print(f"Finished with total reward: {total_reward}")
            break


def plot_state_values(maze, q_table, grid_size, title="State Values"):
    """
    Visualize the state values V(s) in each walkable square using a dictionary.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

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
                state_value = state_values.get(state, 0.00)  # Default to 0.00 if state is missing
                ax.text(col + 0.5, grid_size[0] - row - 0.5, f"{state_value:.2f}", ha="center", va="center", fontsize=12, color="blue")

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
    plt.show()


if __name__ == "__main__":
    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

    maze = np.array(env.desc) 

    initial_q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    state_values = {state: 0.00 for state in range(env.observation_space.n)}
    plot_state_values(maze,state_values, grid_size=(maze.shape[0], maze.shape[1]), title="State Values Before Training")


    # Hyperparameters
    episodes = 1000
    gamma = 0.99 # Discount factor
    epsilon = 1.0 # Exploration rate
    epsilon_decay = 0.999 # Decay
    
    # Train Monte Carlo
    q_table_mc, mc_time, mc_rewards = monte_carlo(env, episodes, gamma, epsilon, epsilon_decay)

    state_values = {state: max(q_table_mc[state]) for state in q_table_mc.keys()}
    plot_state_values(maze, state_values, grid_size=(maze.shape[0], maze.shape[1]), title="State Values After Training")

    # Visualize learned policies
    print("\n--- Visualizing Monte Carlo Policy ---")
    visualize_policy(env, q_table_mc)

    # Summary of results
    print("\nMonte Carlo Training Time:")
    print(f"Total time: {mc_time:.2f} seconds")

    ## Plotting of rewards
    ## to ensure the lengths of x-axis and y-axis data match
    if len(range(episodes)) != len(mc_rewards):
        print(f"Mismatch: x has {len(range(episodes))} points, y has {len(mc_rewards)} points.")
    else:
        plt.plot(range(len(mc_rewards)), mc_rewards, label="Rewards", color='blue')
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("Reward Progression Over Episodes")
        plt.legend()
        plt.show()
