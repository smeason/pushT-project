import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from torch.distributions import Normal
import time
import gymnasium as gym
import gym_pusht
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import GrayscaleObservation
from gymnasium import ActionWrapper, spaces

from param import params
from agents import ActorCritic, ActorCriticCNN, ActorCriticDiscreteWithHistory, ActorCriticWithHistory


class RolloutBuffer:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.next_states = []
        self.durations = []
        self.terminateds = []
        self.truncateds = []

    def __len__(self) -> int:
        return len(self.actions)

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.next_states.clear()
        self.durations.clear()
        self.terminateds.clear()
        self.truncateds.clear()

    # def sample_batch(self, batch_size: int) -> Batch:
    #     """Samples transitions uniformly at random, with replacement.

    #     Args:
    #         batch_size: Number of transitions.

    #     Returns:
    #         The transitions, stacked into tensors.
    #     """
    #     transitions = random.choices(self._transitions, k=batch_size)
    #     return Batch(
    #         states=torch.stack([tr.obs for tr in transitions]),
    #         actions=torch.tensor([[tr.act] for tr in transitions]),
    #         rewards=torch.tensor([[tr.reward] for tr in transitions], dtype=torch.float32),
    #         next_states=torch.stack([tr.next_obs for tr in transitions]),
    #         dones=torch.tensor([[tr.done] for tr in transitions]),
    #     )


class PPO:
    def __init__(
        self,
        env,
        hp: params,
        policy_type: str = "mlp",
        state_dim: int = 5,
        action_dim: int = 2,
        frame_stack_size: int = 4,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.0003,
        max_ppo_steps: int = 100,
        max_episode_len: int = 200,
        clip: float = 0.2,
        discount_gamma: float = 0.99,
        gae_lambda: float = 0.90,
        update_epochs: int = 10,
        batch_size: int = 20,
        entropy_coefficient: float = 0.01,
        action_every_n_steps: int = 4,
        discrete_actions: bool = False,
    ) -> None:
        self.env = env
        self.hp = hp
        self.policy_type = policy_type
        self.state_dim = state_dim if state_dim is not None else env.observation_space.shape[0]
        self.action_dim = action_dim if action_dim is not None else env.action_space.shape[0]
        self.frame_stack_size = frame_stack_size
        self.max_steps = max_ppo_steps
        self.MAX_EPISODE_LEN = max_episode_len
        self.clip = clip
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.num_rollouts = batch_size
        self.entropy_coefficient = entropy_coefficient
        if action_every_n_steps < 1:
            raise ValueError("action_every_n_steps must be at least 1")
        self.action_every_n_steps = action_every_n_steps
        self.discrete_actions = discrete_actions
        self.buffer = RolloutBuffer()
        self.last_success_count = 0
        self.last_avg_contacts = 0.0

        if self.policy_type == "mlp":
            if self.discrete_actions:
                self.policy = ActorCriticDiscreteWithHistory(self.state_dim*self.frame_stack_size, self.action_dim, num_bins=11)
                self.optimizer = torch.optim.Adam([
                    {"params": self.policy.actor.parameters(), "lr": lr_actor},
                    {"params": self.policy.critic.parameters(), "lr": lr_critic},
                ])
            else:
                self.policy = ActorCriticWithHistory(self.state_dim*self.frame_stack_size, self.action_dim)
                self.optimizer = torch.optim.Adam([
                    {"params": self.policy.actor.parameters(), "lr": lr_actor},
                    {"params": self.policy.critic.parameters(), "lr": lr_critic},
                    {"params": self.policy.log_std, "lr": lr_actor},
                ])
        elif self.policy_type == "conv":
            self.policy = ActorCriticCNN(in_channels=self.frame_stack_size, action_dim=self.action_dim, action_std=hp.action_std)
            self.optimizer = torch.optim.Adam([
                {"params": self.policy.feature_extractor.parameters(), "lr": lr_actor},
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_net.parameters(), "lr": lr_critic},
                {"params": self.policy.log_std, "lr": lr_actor},
            ])
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.eval()

        self.loss_fn = nn.MSELoss()

    def _to_environment_action(self, action: torch.Tensor) -> np.ndarray:
        # convert raw action tensor output to usable action numpy array
        action_array = action.detach().cpu().numpy()
        if action_array.ndim == 2:
            action_array = action_array[0]

        if self.discrete_actions:
            # Map indices 0-10 to floats -1.0 to +1.0 (steps of 0.2)
            action_array = (action_array / 5.0) - 1.0

        low = np.asarray(self.env.action_space.low, dtype=np.float32)
        high = np.asarray(self.env.action_space.high, dtype=np.float32)
        scaled = low + (high - low) * (np.clip(action_array, -1.0, 1.0) + 1.0) / 2.0
        return scaled.astype(np.float32)

    def collect_dataset(self):
        self.buffer.clear()
        self.policy_old.eval()

        capture_rollout = getattr(self, "capture_rollout", True)
        start_pose = None
        end_pose = None
        trajectory = []
        total_rewards = []
        episode_lengths = []
        successful_runs = 0
        total_contacts = 0

        for i in range(self.num_rollouts):
            observation, _ = self.env.reset()
            if i == 0 and capture_rollout:
                start_pose = self.env.render()

            ep_reward = 0
            ep_len = 0
            ep_contacts = 0
            ep_success = False
            was_in_contact = False

            while ep_len < self.MAX_EPISODE_LEN:
                state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                if self.policy_type == 'conv':
                    state = state / 255.0

                with torch.no_grad():
                    action, log_prob, state_value = self.policy_old.act(state)

                environment_action = self._to_environment_action(action)
                macro_reward = 0.0
                duration = 0
                steps_to_run = min(self.action_every_n_steps, self.MAX_EPISODE_LEN - ep_len)

                terminated = False
                truncated = False
                for j in range(steps_to_run):
                    next_observation, reward, terminated, truncated, info = self.env.step(environment_action)
                    if i == 0 and j == 0 and ep_len % 100 == 0:
                        print(f'dist_reward {info["dist_reward"]:.4f} | block_goal_dist: {info["block_goal_dist"]:.4f} | angle_reward: {info["angle_reward"]:.4f} | episode_end_reward: {info["episode_end_reward"]:.4f}')
                    done = terminated or truncated

                    macro_reward += (self.discount_gamma ** j) * float(reward)
                    ep_reward += float(reward)
                    ep_len += 1
                    in_contact = bool(info.get("contact", False))
                    ep_contacts += int(in_contact and not was_in_contact)
                    was_in_contact = in_contact
                    ep_success = ep_success or float(info.get("episode_end_reward", 0.0)) > 0 or bool(info.get("is_success", False))
                    duration += 1

                    if i == 0 and capture_rollout:
                        # Get agent pos from info instead of observation
                        trajectory.append(np.asarray(info["pos_agent"]).copy())

                    observation = next_observation

                    if done:
                        break

                if ep_len >= self.MAX_EPISODE_LEN and not terminated:
                    truncated = True

                next_state = torch.as_tensor(observation, dtype=torch.float32).detach().clone()
                if self.policy_type == 'conv':
                    next_state = next_state / 255.0

                self.buffer.states.append(state.squeeze(0).detach().clone())
                self.buffer.actions.append(action.squeeze(0).detach().clone())
                self.buffer.logprobs.append(log_prob.squeeze(0).detach().clone())
                self.buffer.rewards.append(macro_reward)
                self.buffer.state_values.append(state_value.squeeze(0).detach().clone())
                self.buffer.next_states.append(next_state)
                self.buffer.durations.append(duration)
                self.buffer.terminateds.append(bool(terminated))
                self.buffer.truncateds.append(bool(truncated))

                if terminated or truncated:
                    break

            if i == 0 and capture_rollout:
                end_pose = self.env.render()

            total_rewards.append(ep_reward)
            episode_lengths.append(ep_len)
            total_contacts += ep_contacts
            successful_runs += int(ep_success)

        self.last_success_count = successful_runs
        self.last_completed_run_count = len(episode_lengths)
        self.last_avg_contacts = total_contacts / max(len(episode_lengths), 1)
        return (self.buffer, float(np.mean(total_rewards)), float(np.mean(episode_lengths)),
            start_pose, end_pose, trajectory)

    def update_policy_clip(self):

        states = torch.stack(self.buffer.states).detach()
        actions = torch.stack(self.buffer.actions).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        values = torch.stack(self.buffer.state_values).detach().float()
        rewards = torch.as_tensor(np.asarray(self.buffer.rewards), dtype=torch.float32)
        next_states = torch.stack(self.buffer.next_states).detach()
        is_vector = rewards.ndim == 2
        with torch.no_grad():
            critic_states = next_states.flatten(0, 1) if is_vector else next_states
            next_values = self.policy.critic(critic_states).squeeze(-1).reshape(rewards.shape)
        durations = torch.as_tensor(np.asarray(self.buffer.durations), dtype=torch.float32)
        terminateds = torch.as_tensor(np.asarray(self.buffer.terminateds), dtype=torch.bool)
        episode_ends = terminateds | torch.as_tensor(np.asarray(self.buffer.truncateds), dtype=torch.bool)

        # calculate gae - generalized advantage estimate
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(rewards[0])
        for t in reversed(range(len(rewards))):
            bootstrap_mask = 1.0 - terminateds[t].float()
            trace_mask = 1.0 - episode_ends[t].float()
            delta = (
                rewards[t]
                + self.discount_gamma ** durations[t] * next_values[t] * bootstrap_mask
                - values[t]
            )
            gae = delta + (
                (self.discount_gamma * self.gae_lambda) ** durations[t]
                * trace_mask
                * gae
            )
            advantages[t] = gae

        returns = advantages + values
        if is_vector:
            states = states.flatten(0, 1)
            actions = actions.flatten(0, 1)
            old_logprobs = old_logprobs.flatten(0, 1)
            returns = returns.flatten(0, 1)
            advantages = advantages.flatten(0, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # update policy with gradient ascent
        for _ in range(self.update_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - old_logprobs)
            surrogate_one = ratios * advantages
            surrogate_two = torch.clamp(
                ratios, 1.0 - self.clip, 1.0 + self.clip
            ) * advantages

            policy_loss = -torch.min(surrogate_one, surrogate_two).mean()
            value_loss = self.loss_fn(state_values, returns)
            entropy_loss = -self.entropy_coefficient * entropy.mean()
            loss = policy_loss + 0.5 * value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def run(self, ax=None):
        x_data = []
        y_data = []

        for i in range(self.max_steps):
            progress = i / max(self.max_steps - 1, 1)
            # linear decay
            weight = self.hp.initial_agent_object_weight + progress * (
                self.hp.final_agent_object_weight - self.hp.initial_agent_object_weight
            )
            self.entropy_coefficient = self.entropy_coefficient + progress * (self.hp.entropy_coefficient_min - self.entropy_coefficient)
            if isinstance(self.env, (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv)):
                self.env.set_attr("agent_object_weight", weight)
            elif not self.env.set_wrapper_attr("agent_object_weight", weight, force=False):
                raise AttributeError("Environment has no agent_object_weight attribute")

            self.capture_rollout = (i + 1) % 3 == 0
            start_time = time.perf_counter()
            _, avg_reward, avg_ep_len, start_pose, end_pose, trajectory = self.collect_dataset()
            end_time = time.perf_counter()
            completed_runs = getattr(self, "last_completed_run_count", self.num_rollouts)
            print(
                f'Step {i:03d}/{self.max_steps} | Avg Reward: {avg_reward:.4f} '
                f'| Avg Episode Length: {avg_ep_len:.1f} | Distance Weight: {weight:.4f} '
                f'| Successful Runs: {self.last_success_count}/{completed_runs} '
                f'| Avg Block Contacts/Run: {self.last_avg_contacts:.2f} '
                f'| collect time: {(end_time - start_time):.2f} s'
            )

            x_data.append(i)
            y_data.append(avg_reward)

            # if ax is not None:
            #     ax.clear()
            #     ax.plot(x_data, y_data, color='tab:blue', linewidth=2)
            #     ax.set_title("Avg Reward")
            #     ax.set_xlabel("Step")
            #     ax.set_ylabel("Avg. Reward")
            #     ax.grid(True)

            #     # 3. Refresh the Colab cell output smoothly
            #     clear_output(wait=True)
            #     display(fig)

            rewards = self.buffer.rewards.copy()
            if (i + 1) % 3 == 0 and start_pose is not None and end_pose is not None and len(trajectory) > 0:
                self.save_rollout_as_jpg(start_pose, end_pose, trajectory, i, rewards)

            start_time = time.perf_counter()
            self.update_policy_clip()
            end_time = time.perf_counter()
            print(f'update policy in {(end_time - start_time):.2f} s')

        return x_data, y_data

    def act(self, state):
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        if self.policy_type == 'conv':
            state = state / 255.0
        with torch.inference_mode():
            action, log_prob, state_value = self.policy_old.act(state)
        return action, log_prob, state_value

    def save_rollout_as_jpg(self, start_pose, end_pose, trajectory, step, rewards):
        overlay = np.clip(0.5*start_pose + 0.5 * end_pose, 0, 255).astype(np.uint8)
        trajectory = np.asarray(trajectory, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if rewards.size == 0:
            raise ValueError("Cannot plot a reward histogram with no collected rewards")

        fig, (ax, reward_ax) = plt.subplots(
            1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.25, 1]}
        )
        ax.imshow(overlay, extent=(0, 512, 512, 0))
        ax.plot(trajectory[:,0], trajectory[:,1], color="blue", linewidth=2, label="Agent path")
        ax.scatter(*trajectory[0], color="yellow", s=30, label="Agent start")
        ax.scatter(*trajectory[-1], color="red", s=30, label="Agent end")

        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.set_aspect('equal')
        ax.set_title(f'Rollout {step}')

        reward_ax.hist(rewards, bins="auto", color="tab:orange", edgecolor="black")
        reward_ax.set_title(f'Reward Distribution - PPO Iteration {step}')
        reward_ax.set_xlabel('Collected reward per PPO action')
        reward_ax.set_ylabel('Count')
        reward_ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(f"rollout_square{step:03d}.jpg")
        plt.close(fig)

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.policy_old.load_state_dict(state_dict)
        self.policy_old.eval()
        self.policy.load_state_dict(state_dict)
