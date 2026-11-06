import gymnasium as gym
import gym_pusht
import numpy as np
import torch
import torch.nn as nn

class PPO:

    def __init__(self, env, policy_type:str = 'mlp') -> None:
        self.env = env
        self.theta = 0
        self.value = 0
        self.max_steps = 100
        self.MAX_EPISODE_LEN = 500
        self.clip = 0.2
        self.discount_gamma = 0.9

        self.num_inputs = 5
        self.num_outputs = 2 # action space is target position, x and y continuous in [0, 512]
        # self.num_outputs = 2 # action space size
        
        self.actor: nn.Sequential = nn.Sequential(
            nn.Linear(self.num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_outputs),
        )
        if policy_type == "mlp":
            self.actor = nn.Sequential(
                nn.Linear(self.num_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_outputs)
            )
        elif policy_type == "conv":
            print(f'conv policy not implemented yet')
            raise NotImplementedError()

        # maps observation to scalar V(s). predicts returns
        self.critic = nn.Sequential(
            nn.Linear(self.num_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_outputs)
        )


        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)


    def run(self):

        for k in range(self.max_steps):
            trajectories = self.collect_dataset(self.actor, 20, k)

            rewards_to_go = 0
            for i in range(len(trajectories)):
                rewards = []
                G_t = 0
                for (obs, action, reward, done) in trajectories[i]:
                    G_t = reward + self.discount_gamma*G_t
                    rewards.append(G_t)

            # use Generalized Advantage Estimation (GAE)
            timesteps = len(trajectories[0])
            advantage = np.zeros(10) # size of num timesteps
            td_errors = []
            lambda_ = 0.95
            for (obs, action, reward, done) in trajectories[0]:
                td_error = reward + self.discount_gamma * self.value(s_t+1) - self.value(s_t)
                td_errors.append(td_error)
            
            for i, (obs, action, reward, done) in enumerate(reversed(trajectories[0])):
                t = timesteps-t
                advantage[t] = td_error[t] + self.discount_gamma * lambda_ * advantage[t+1]


            self.update_policy_clip()

            # fit value function by regression on mean-squared error with gradient descent
            self.value = np.argmin([])


    def update_policy_clip(self):
        """Update with stochastic gradient ascent with Adam"""

        self.actor = None

    def collect_dataset(self, policy: nn.Sequential, num_rollouts: int = 20, k):
        D = []
        for i in range(num_rollouts):
            history = []
            observation, info = self.env.reset()
            for _ in range(self.MAX_EPISODE_LEN):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    mean = self.actor(obs_tensor)
                    std = self.log_std.exp().expand_as(mean)
                    dist = torch.distributions.Normal(mean, std)

                    action = dist.sample()
                    old_log_prob = dist.log_prob(action).sum(dim=-1)
                    value = self.critic(obs_tensor).squeeze(-1)

                action_np = action.squeeze(0).cpu().numpy()

                # action = policy(observation)
                observation, reward, terminated, truncated, info = self.env.step(action_np)
                # image = self.env.render()
                history.append((observation, action, reward, terminated or truncated))

                if terminated or truncated:
                    # observation, info = self.env.reset()
                    break

            D.append(history)
        return D

    