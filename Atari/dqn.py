import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports all our hyperparameters from the other file
from hyperparams import Hyperparameters as params

# stable_baselines3 have wrappers that simplify
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer


# Creates our gym environment with all our wrappers.
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            video_path = f"videos/{run_name}"
            os.makedirs(video_path, exist_ok=True)  # Ensure directory exists
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, video_path)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"

    random.seed(params.seed)
    rng = np.random.default_rng(params.seed)  # Updated seeding
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # We'll be using experience replay memory for training our DQN.
    # It stores the transitions that the agent observes, allowing us to reuse this data later.
    # By sampling from it randomly, the transitions that build up a batch are decorrelated.
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs = envs.reset()
    for global_step in range(params.total_timesteps):
        # Here we get epsilon for our epsilon greedy.
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps,
                                  global_step)

        if random.random() < epsilon:
            actions = envs.action_space.sample()  # Sample a random action from the environment
        else:
            q_values = q_network(torch.tensor(obs, device=device, dtype=torch.float32))  # Get q_values from the network
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Take a step in the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Here we print our reward.
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Here we store the transitions in D
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs

        # Training
        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                # Sample random minibatch of transitions from D
                data = rb.sample(params.batch_size)

                with torch.no_grad():
                    # Calculate the target Q value
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards + params.gamma * target_max * (1 - data.dones)

                # Ensure actions are of shape [batch_size, 1]
                actions = data.actions.view(-1, 1)  # Correct the shape of actions
                q_values = q_network(data.observations)

                old_val = q_values.gather(1, actions).squeeze()  # Remove the extra dimension after gather

                loss = F.mse_loss(old_val, td_target)
                # Perform gradient descent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if global_step % params.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        params.tau * q_network_param.data + (1.0 - params.tau) * target_network_param.data
                    )

    if params.save_model:
        # Create the directory if it doesn't exist
        model_dir = f"C:/Users/thoma/MasterProjects/AtariGame/{run_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, f"{params.exp_name}_model")
        torch.save(q_network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    envs.close()