import numpy as np
import random
import time
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def initialize_q_table(states, actions):
    """Initialize the Q-table with small random values."""
    return np.random.uniform(low=-0.01, high=0.01, size=(states, actions))

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


if __name__ == "__main__":
    # Create FrozenLake environment
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

    # Hyperparameters
    episodes = 1000
    alpha = 0.4  # Learning rate
    gamma = 0.6  # Discount factor
    epsilon = 0.4  # Exploration rate
    epsilon_decay = 0.995

    # Train SARSA
    q_table,rewards = sarsa(env, episodes, alpha, gamma, epsilon, epsilon_decay)

    # Visualize the learned policy
    print("\n--- Visualizing SARSA Policy ---")
    visualize_sarsa_policy(env, q_table)

    episodes=np.linspace(1,episodes,episodes)
    training_data={'episodes':episodes,'rewards':rewards}
    data=pd.DataFrame(training_data)

    #Plotting training information
    plt.plot(episodes,rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()
    