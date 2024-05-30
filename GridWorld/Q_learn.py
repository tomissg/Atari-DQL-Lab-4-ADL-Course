import matplotlib.pyplot as plt
import numpy as np
import random
def plot_gridworld(env, title="Gridworld"):
    """
    Plot the current state of the Gridworld.
    """
    grid = np.zeros(env.size)
    grid[env.goal] = 10
    for obs in env.obstacles:
        grid[obs] = -10
    grid[env.state] = 1

    plt.imshow(grid, cmap='viridis', interpolation='none')
    plt.title(title)
    plt.colorbar()

    # Plot the agent's path
    positions = np.array(env.position_history)
    if len(positions) > 0:
        plt.plot(positions[:, 1], positions[:, 0], marker='o', color='red', markersize=5, linestyle='-')

    plt.show()


def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, plot_interval=100):
    """
    Q-learning algorithm to find the optimal policy.
    """
    q_table = {}
    for row in range(env.size[0]):
        for col in range(env.size[1]):
            state = (row, col)
            q_table[state] = {action: 0 for action in env.get_actions()}

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_actions())  # Explore
            else:
                action = max(q_table[state], key=q_table[state].get)  # Exploit

            next_state, reward, done = env.step(action)
            next_max = max(q_table[next_state].values())

            # Q-learning update
            q_table[state][action] += alpha * (reward + gamma * next_max - q_table[state][action])
            state = next_state

        if episode % plot_interval == 0:
            plot_gridworld(env, title=f"Episode {episode}")

    return q_table


def extract_policy(q_table):
    """
    Extract the optimal policy from the Q-table.
    """
    policy = {}
    for state in q_table:
        policy[state] = max(q_table[state], key=q_table[state].get)
    return policy
