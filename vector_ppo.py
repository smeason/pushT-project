import gymnasium as gym
import numpy as np
import torch

from ppo import PPO


class VectorPPO(PPO):
    """PPO with independent, automatically reset parallel environments."""

    def __init__(self, vec_env, rollout_steps=128, seed=None, **kwargs):
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be at least 1")
        if vec_env.metadata.get("autoreset_mode") != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError("VectorPPO requires SAME_STEP autoreset")
        super().__init__(vec_env, **kwargs)
        self.num_envs = vec_env.num_envs
        self.rollout_steps = rollout_steps
        self.seed = seed
        self.observations = None
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_contact_counts = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_successes = np.zeros(self.num_envs, dtype=np.bool_)
        self.episode_was_in_contact = np.zeros(self.num_envs, dtype=np.bool_)

    def collect_dataset(self):
        self.buffer.clear()
        self.policy_old.eval()
        if self.observations is None:
            self.observations, _ = self.env.reset(seed=self.seed)

        completed_rewards = []
        completed_lengths = []
        completed_contacts = []
        completed_successes = []
        capture_rollout = getattr(self, "capture_rollout", True)
        start_frames = self.env.render() if capture_rollout else None
        start_pose = start_frames[0] if start_frames is not None and len(start_frames) > 0 else None
        trajectory = []
        trajectory_episode_finished = False
        for _ in range(self.rollout_steps):
            states = torch.as_tensor(self.observations, dtype=torch.float32).clone()
            if self.policy_type == "conv":
                states = states / 255.0
            with torch.no_grad():
                actions, log_probs, state_values = self.policy_old.act(states)

            next_observations, rewards, terminateds, truncateds, infos = self.env.step(
                self._to_environment_action_batched(actions)
            )
            dones = np.logical_or(terminateds, truncateds)
            positions = infos.get("pos_agent")
            position_mask = infos.get("_pos_agent")
            if capture_rollout and not trajectory_episode_finished:
                if dones[0]:
                    final_info = infos.get("final_info", {})
                    final_info_mask = infos.get("_final_info", np.zeros(self.num_envs, dtype=np.bool_))
                    final_pos_mask = final_info.get("_pos_agent", final_info_mask)
                    final_positions = final_info.get("pos_agent")
                    if (
                        final_positions is not None
                        and final_info_mask[0]
                        and final_pos_mask[0]
                    ):
                        trajectory.append(np.asarray(final_positions[0]).copy())
                    trajectory_episode_finished = True
                elif positions is not None and (position_mask is None or position_mask[0]):
                    trajectory.append(np.asarray(positions[0]).copy())
            # SAME_STEP returns reset observations for completed environments.
            # Use the final observations for the transition and bootstrapping.
            transition_next = np.array(next_observations, copy=True)
            if np.any(dones):
                final_mask = infos.get("_final_obs")
                if final_mask is None or not np.all(final_mask[dones]):
                    raise RuntimeError("Missing final observations for completed environments")
                for i in np.flatnonzero(dones):
                    transition_next[i] = infos["final_obs"][i]

            durations = np.ones(self.num_envs, dtype=np.int64)
            raw_rewards = np.asarray(rewards, dtype=np.float64).copy()
            if "action_repeat_steps" in infos:
                durations[~dones] = infos["action_repeat_steps"][~dones]
                raw_rewards[~dones] = infos["undiscounted_reward"][~dones]
            if np.any(dones) and "final_info" in infos:
                final_info = infos["final_info"]
                if "action_repeat_steps" in final_info:
                    durations[dones] = final_info["action_repeat_steps"][dones]
                    raw_rewards[dones] = final_info["undiscounted_reward"][dones]

            self.episode_rewards += raw_rewards
            self.episode_lengths += durations
            contact_counts = infos.get("contact_events", np.zeros(self.num_envs, dtype=np.int64))
            success_rewards = infos.get("episode_end_reward", np.zeros(self.num_envs))
            env_successes = infos.get("is_success", np.zeros(self.num_envs, dtype=np.bool_))
            final_info = infos.get("final_info", {})
            final_info_mask = infos.get("_final_info", np.zeros(self.num_envs, dtype=np.bool_))
            for env_index in range(self.num_envs):
                contacts = int(contact_counts[env_index])
                success = bool(success_rewards[env_index] > 0 or env_successes[env_index])
                if dones[env_index] and final_info_mask[env_index]:
                    if "contact_events" in final_info and final_info.get("_contact_events", final_info_mask)[env_index]:
                        contacts = int(final_info["contact_events"][env_index])
                    final_success = final_info.get("episode_end_reward", np.zeros(self.num_envs))
                    final_is_success = final_info.get("is_success", np.zeros(self.num_envs, dtype=np.bool_))
                    success = bool(final_success[env_index] > 0 or final_is_success[env_index])
                if "contact_events" not in infos:
                    in_contact = bool(infos.get("contact", np.zeros(self.num_envs, dtype=np.bool_))[env_index])
                    contacts = int(in_contact and not self.episode_was_in_contact[env_index])
                    self.episode_was_in_contact[env_index] = in_contact
                self.episode_contact_counts[env_index] += contacts
                self.episode_successes[env_index] |= success
                if dones[env_index]:
                    completed_rewards.append(float(self.episode_rewards[env_index]))
                    completed_lengths.append(float(self.episode_lengths[env_index]))
                    completed_contacts.append(int(self.episode_contact_counts[env_index]))
                    completed_successes.append(bool(self.episode_successes[env_index]))
                    self.episode_rewards[env_index] = 0
                    self.episode_lengths[env_index] = 0
                    self.episode_contact_counts[env_index] = 0
                    self.episode_successes[env_index] = False
                    self.episode_was_in_contact[env_index] = False

            next_states = torch.as_tensor(transition_next, dtype=torch.float32).clone()
            if self.policy_type == "conv":
                next_states = next_states / 255.0
            self.buffer.states.append(states)
            self.buffer.actions.append(actions.detach().clone())
            self.buffer.logprobs.append(log_probs.detach().clone())
            self.buffer.state_values.append(state_values.detach().clone())
            self.buffer.rewards.append(np.asarray(rewards).copy())
            self.buffer.next_states.append(next_states)
            self.buffer.durations.append(durations)
            self.buffer.terminateds.append(np.asarray(terminateds).copy())
            self.buffer.truncateds.append(np.asarray(truncateds).copy())
            self.observations = np.array(next_observations, copy=True)

        # Keep the background tied to the same episode start. SAME_STEP autoreset
        # may otherwise make the final render belong to a later episode.
        end_pose = start_pose
        avg_reward = float(np.mean(completed_rewards)) if completed_rewards else float(np.mean(self.episode_rewards))
        avg_length = float(np.mean(completed_lengths)) if completed_lengths else float(np.mean(self.episode_lengths))
        self.last_success_count = int(sum(completed_successes))
        self.last_completed_run_count = len(completed_contacts)
        self.last_avg_contacts = float(np.mean(completed_contacts)) if completed_contacts else 0.0
        return self.buffer, avg_reward, avg_length, start_pose, end_pose, trajectory

    def _to_environment_action_batched(self, actions: torch.Tensor) -> np.ndarray:
        actions_np = actions.detach().cpu().numpy()
        if self.discrete_actions:
            actions_np = (actions_np / 5.0) - 1.0
        low = np.asarray(self.env.single_action_space.low, dtype=np.float32)
        high = np.asarray(self.env.single_action_space.high, dtype=np.float32)
        scaled = low + (high - low) * (np.clip(actions_np, -1.0, 1.0) + 1.0) / 2.0
        return scaled.astype(np.float32)
