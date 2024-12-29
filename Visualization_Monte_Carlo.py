import numpy as np
import random
import time
import gym
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
            if episode % 10 == 0:
                env.render()

            if done or truncated:
                print(f"Episode finished with total reward: {total_reward}\n")
                break

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
    return q_table, elapsed_time

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
        time.sleep(1)  # Pause for better visualization

        if done or truncated:
            print(f"Finished with total reward: {total_reward}")
            break

if __name__ == "__main__":
    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

    # Hyperparameters
    episodes = 100
    gamma = 0.99  # Discount factor
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995

    # Train Monte Carlo
    q_table_mc, mc_time = monte_carlo(env, episodes, gamma, epsilon, epsilon_decay)

    # Visualize learned policies
    print("\n--- Visualizing Monte Carlo Policy ---")
    visualize_policy(env, q_table_mc)

    # Summary of results
    print("\nMonte Carlo Training Time:")
    print(f"Total time: {mc_time:.2f} seconds")
