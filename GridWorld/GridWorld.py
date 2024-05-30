import numpy as np
import random


class Gridworld:
    def __init__(self, size=(5, 5), start=(0, 0), goal=(4, 4), num_obstacles=3):
        self.size = size
        self.start = start
        self.goal = goal
        self.num_obstacles = num_obstacles
        self.state = start
        self.actions = ['North', 'South', 'West', 'East']
        self.action_space = len(self.actions)
        self.obstacles = self._generate_obstacles()
        self.position_history = []

    def _generate_obstacles(self):
        """Generate a set of random obstacles avoiding start and goal positions."""
        obstacles = set()
        while len(obstacles) < self.num_obstacles:
            row = random.randint(0, self.size[0] - 1)
            col = random.randint(0, self.size[1] - 1)
            if (row, col) != self.start and (row, col) != self.goal:
                obstacles.add((row, col))
        return list(obstacles)

    def reset(self):
        """Reset the environment to the start state."""
        self.state = self.start
        self.position_history = [self.state]
        return self.state

    def step(self, action):
        """
        Take an action and return the next state, reward, and done.
        """
        row, col = self.state
        if action == 'North':
            row = max(row - 1, 0)
        elif action == 'South':
            row = min(row + 1, self.size[0] - 1)
        elif action == 'West':
            col = max(col - 1, 0)
        elif action == 'East':
            col = min(col + 1, self.size[1] - 1)

        new_state = (row, col)

        if new_state in self.obstacles:
            new_state = self.state  # Hit an obstacle, stay in the same state

        reward = -1  # Default reward for each step
        if new_state == self.goal:
            reward = 10  # Reward for reaching the goal
        self.state = new_state
        self.position_history.append(self.state)
        return new_state, reward, new_state == self.goal

    def get_actions(self):
        """Return the list of possible actions."""
        return self.actions

    def render(self):
        """Render the current state of the Gridworld."""
        grid = np.zeros(self.size)
        grid[self.goal] = 10
        for obs in self.obstacles:
            grid[obs] = -10
        grid[self.state] = 1
        print(grid)