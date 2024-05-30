from GridWorld import Gridworld
from Q_learn import q_learning, extract_policy

# Create the Gridworld environment
env = Gridworld()

# Perform Q-learning
q_table = q_learning(env)

# Extract the optimal policy
policy = extract_policy(q_table)

# Display the learned policy
print("Optimal Policy:")
for state in sorted(policy):
    print(f"State {state}: {policy[state]}")