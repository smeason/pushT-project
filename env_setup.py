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
from typing import cast
from gym_pusht.envs.pusht import PushTEnv
from arbitrary_shape_env import ArbitraryShapePushEnv

from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import FlattenObservation
from gymnasium.wrappers import GrayscaleObservation
from gymnasium import ActionWrapper, spaces
from param import params

SUCCESS_DISTANCE = 10.0 # T 15.0
SUCCESS_ANGLE = np.pi/20 # T np.pi / 18

def make_env(
    hp: params,
    env_id="gym_pusht/PushT-v0",
    max_dist=None,
    max_angle=None,
    seed=0,
    min_dist=None,
    min_angle=None,
    action_repeat=1,
    max_episode_steps=None,
    discount_gamma=0.99,
    block_shape=None,
    block_width=None,
    block_height=None,
    block_vertices=None,
    rotation_symmetry_order=None,
    agent_friction=None,
    block_friction=None,
    wall_friction=None,
    environment_damping=None,
):
    """Factory function to create a single environment instance with all wrappers."""
    def thunk():
        shape_name = str(hp.block_shape if block_shape is None else block_shape).lower()
        if shape_name == "t":
            env: gym.Env = gym.make(
                env_id, render_mode="rgb_array", obs_type="state",
                max_episode_steps=max_episode_steps,
                damping=(
                    hp.environment_damping
                    if environment_damping is None
                    else environment_damping
                ),
            )
        else:
            env = ArbitraryShapePushEnv(
                shape=shape_name,
                width=hp.block_width if block_width is None else block_width,
                height=hp.block_height if block_height is None else block_height,
                vertices=hp.block_vertices if block_vertices is None else block_vertices,
                rotation_symmetry_order=(
                    hp.rotation_symmetry_order
                    if rotation_symmetry_order is None
                    else rotation_symmetry_order
                ),
                render_mode="rgb_array",
                obs_type="state",
                damping=(
                    hp.environment_damping
                    if environment_damping is None
                    else environment_damping
                ),
            )
            if max_episode_steps is not None:
                env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        env = ShapeFrictionWrapper(
            env,
            agent_friction=hp.agent_friction if agent_friction is None else agent_friction,
            block_friction=hp.block_friction if block_friction is None else block_friction,
            wall_friction=hp.wall_friction if wall_friction is None else wall_friction,
        )
        env = DistanceCurriculumAndRewardWrapper(
            env,
            max_distance=hp.initial_max_dist if max_dist is None else max_dist,
            max_angle=np.deg2rad(hp.initial_max_angle if max_angle is None else max_angle),
            agent_object_weight=hp.initial_agent_object_weight,
            agent_object_progress_weight=hp.agent_object_progress_weight,
            object_goal_progress_weight=hp.object_goal_progress_weight,
            angle_progress_weight=hp.angle_progress_weight,
            contact_reward_decay=hp.contact_reward_decay,
            discount_gamma=hp.discount_gamma,
            min_distance=hp.initial_min_dist if min_dist is None else min_dist,
            min_angle=np.deg2rad(hp.initial_min_angle if min_angle is None else min_angle),
        )
        env = PushTVelocityWrapper(env, max_speed=hp.max_speed, min_speed=hp.min_speed)
        env = ActionSmoothnessPenaltyWrapper(env, penalty_weight=hp.smoothness_penalty_weight)
        env = NormalizeObservation(env)
        env = FrameStackObservation(env, stack_size=hp.frame_stack_size)
        env = FlattenObservation(env)
        env = ActionRepeatWrapper(env, action_repeat, discount_gamma)
        env.action_space.seed(seed)
        return env
    return thunk



class ActionRepeatWrapper(gym.Wrapper):
    """Hold an action for several steps and report the actual duration."""
    def __init__(self, env, repeat, discount_gamma):
        super().__init__(env)
        if repeat < 1:
            raise ValueError("action repeat must be at least 1")
        self.repeat = repeat
        self.discount_gamma = discount_gamma
        self._was_in_contact = False

    def reset(self, **kwargs):
        self._was_in_contact = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        discounted_reward = float(reward)
        total_reward = float(reward)
        contact_events = int(bool(info.get("contact", False)) and not self._was_in_contact)
        self._was_in_contact = bool(info.get("contact", False))
        duration = 1
        while duration < self.repeat and not (terminated or truncated):
            obs, reward, terminated, truncated, info = self.env.step(action)
            discounted_reward += self.discount_gamma ** duration * float(reward)
            total_reward += float(reward)
            in_contact = bool(info.get("contact", False))
            contact_events += int(in_contact and not self._was_in_contact)
            self._was_in_contact = in_contact
            duration += 1
        info = dict(info)
        info["contact_events"] = contact_events
        info["action_repeat_steps"] = duration
        info["undiscounted_reward"] = total_reward
        return obs, discounted_reward, terminated, truncated, info


class ShapeFrictionWrapper(gym.Wrapper):
    """Set friction coefficients on the Pymunk collision shapes after reset."""

    def __init__(self, env, agent_friction=None, block_friction=None, wall_friction=None):
        super().__init__(env)
        self.agent_friction = self._validate_friction("agent_friction", agent_friction)
        self.block_friction = self._validate_friction("block_friction", block_friction)
        self.wall_friction = self._validate_friction("wall_friction", wall_friction)

    @staticmethod
    def _validate_friction(name, value):
        if value is None:
            return None
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number")
        return float(value)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        base = cast(PushTEnv, self.unwrapped)
        for shape in base.space.shapes:
            if shape.body is base.agent:
                friction = self.agent_friction
            elif shape.body is base.block:
                friction = self.block_friction
            else:
                friction = self.wall_friction
            if friction is not None:
                shape.friction = friction
        return observation, info


### Wrap pushT environment to use a velocity-based action space (instead of position)
class PushTVelocityWrapper(ActionWrapper):
    def __init__(self, env, max_speed=40.0, min_speed=5.0):
        super().__init__(env)

        self.max_speed = max_speed
        self.min_speed = min_speed
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        base = cast(PushTEnv, self.unwrapped)
        current_position = np.asarray(base.agent.position, dtype=np.float32)
        # print(f'current position {current_position}')

        velocity = np.clip(action, -1.0, 1.0) * self.max_speed
        speed = np.linalg.norm(velocity)
        # print(f'velocity {velocity}')

        # Floor nonzero commands; leave a zero command as a stop.
        if 1e-6 < speed < self.min_speed:
            velocity *= self.min_speed / speed
        # print(f'{self.unwrapped.control_hz}')
        target_position = current_position + velocity / base.control_hz

        action_space = cast(spaces.Box, self.env.action_space)
        low = action_space.low
        high = action_space.high
        # print(f'{np.clip(target_position, low, high)}')
        return np.clip(target_position, low, high).astype(np.float32)

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32).copy()

        # Assuming the observation is:
        # [pusher_x, pusher_y, block_x, block_y, block_angle]

        obs[:4] = (obs[:4] / 256.0) - 1.0
        obs[4] = obs[4] / np.pi

        # Represent angle as sin/cos instead of raw angle
        # angle = obs[4]
        # obs[4] = np.sin(angle)
        # (For a 2D sin/cos representation, you'd expand the observation.)

        return obs


class DistanceCurriculumAndRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,#: gym.Env,
        max_distance=30,
        max_angle=np.deg2rad(15),
        max_attempts=20_000,
        agent_object_weight=0.1,
        object_goal_weight=1.0,
        object_goal_angle_weight=1.0,
        contact_bonus=0.01,
        distance_scale=512.0,
        min_distance=15.0,
        min_angle=np.deg2rad(10),
        agent_object_progress_weight=1.0,
        object_goal_progress_weight=1.0,
        angle_progress_weight=0.25,
        contact_reward_decay=0.5,
        discount_gamma=0.99,
    ):
        super().__init__(env)
        self.agent_object_weight = agent_object_weight
        self.agent_object_progress_weight = agent_object_progress_weight
        self.object_goal_progress_weight = object_goal_progress_weight
        self.angle_progress_weight = angle_progress_weight
        if not np.isfinite(contact_reward_decay) or not 0 <= contact_reward_decay <= 1:
            raise ValueError("contact_reward_decay must be between 0 and 1")
        self.contact_reward_decay = contact_reward_decay
        self._contact_events_this_episode = 0
        if not 0 <= discount_gamma <= 1:
            raise ValueError("discount_gamma must be between 0 and 1")
        self.discount_gamma = discount_gamma
        self.object_goal_weight = object_goal_weight
        self.object_goal_angle_weight = object_goal_angle_weight
        self.contact_bonus = contact_bonus
        self.distance_scale = distance_scale
        if not all(np.isfinite(v) for v in (min_distance, max_distance, min_angle, max_angle)):
            raise ValueError("Curriculum limits must be finite")
        if not 0 <= min_distance <= max_distance:
            raise ValueError("Distance limits must satisfy 0 <= min_distance <= max_distance")
        if not 0 <= min_angle <= max_angle <= np.pi:
            raise ValueError("Angle limits must satisfy 0 <= min_angle <= max_angle <= pi")
        if max_distance < SUCCESS_DISTANCE and max_angle < SUCCESS_ANGLE:
            raise ValueError("Curriculum limits only allow states inside the success threshold")
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_attempts = max_attempts

    def _metrics(self, info):
        agent_xy = np.asarray(info["pos_agent"][:2])
        object_xy = np.asarray(info["block_pose"][:2])
        goal_xy = np.asarray(info["goal_pose"][:2])
        angle_delta = info["block_pose"][2] - info["goal_pose"][2]

        # Check the pusher's shape against the block's shapes.
        base = cast(PushTEnv, self.unwrapped)
        symmetry_order = getattr(base, "rotation_symmetry_order", 1)
        angle_difference = float(
            np.arctan2(
                np.sin(symmetry_order * angle_delta),
                np.cos(symmetry_order * angle_delta),
            ) / symmetry_order
        )
        contact = any(
            hit.shape is not None and hit.shape.body is base.block
            for agent_shape in base.agent.shapes
            for hit in base.space.shape_query(agent_shape)
        )

        # Pymunk's point query gives the signed distance from the pusher center
        # to each block collision shape. Subtract the pusher radius to get the
        # physical clearance between the pusher's surface and the block edge.
        query_point = (float(agent_xy[0]), float(agent_xy[1]))
        block_edge_distance = min(
            shape.point_query(query_point).distance for shape in base.block.shapes
        )
        agent_radius = max(
            (float(getattr(shape, "radius", 0.0)) for shape in base.agent.shapes),
            default=0.0,
        )
        agent_object_distance = max(0.0, float(block_edge_distance) - agent_radius)

        return {
            "agent_object_distance": agent_object_distance,
            "object_goal_distance": float(np.linalg.norm(object_xy - goal_xy)),
            "object_goal_angle_difference": angle_difference,
            "object_goal_angle_error": abs(angle_difference),
            "contact": bool(contact),
        }

    def reset(self, *, seed=None, options=None):
        # Initialize the environment and its RNG before sampling curriculum states.
        self.env.reset(seed=seed)
        base = cast(PushTEnv, self.env.unwrapped)

        for _ in range(self.max_attempts):
            distance = base.np_random.uniform(self.min_distance, self.max_distance)
            angle_error = base.np_random.uniform(self.min_angle, self.max_angle)
            if base.np_random.uniform(0, 1) < 0.5:
                angle_error *= -1

            theta = base.np_random.uniform(0, 2 * np.pi)
            goal_x, goal_y, goal_angle = base.goal_pose
            new_block_x = np.clip(goal_x + distance * np.cos(theta), 80, 432)
            new_block_y = np.clip(goal_y + distance * np.sin(theta), 80, 432)
            new_block_angle = goal_angle + angle_error

            # Keep the agent away from the sampled block to avoid initial overlap.
            agent_x, agent_y = base.agent.position
            for _ in range(100):
                if np.hypot(new_block_x - agent_x, new_block_y - agent_y) > 100:
                    break
                agent_x, agent_y = base.np_random.uniform(80, 432, size=2)
            else:
                continue

            # PushT sets position before angle, shifting the origin around the
            # center of gravity. Compensate for that shift before resetting.
            cx, cy = base.block.center_of_gravity
            sin_angle, cos_angle = np.sin(new_block_angle), np.cos(new_block_angle)
            reset_block_x = new_block_x - (cx * (1 - cos_angle) + cy * sin_angle)
            reset_block_y = new_block_y - (cy * (1 - cos_angle) - cx * sin_angle)
            obs, info = self.env.reset(options={"reset_to_state": [
                agent_x, agent_y, reset_block_x, reset_block_y, new_block_angle,
            ]})
            base = cast(PushTEnv, self.env.unwrapped)
            block_xy = np.asarray(base.block.position)
            goal_xy = np.asarray(base.goal_pose[:2])
            actual_distance = float(np.linalg.norm(block_xy - goal_xy))
            angle_delta = base.block.angle - base.goal_pose[2]
            symmetry_order = getattr(base, "rotation_symmetry_order", 1)
            actual_angle_error = float(abs(
                np.arctan2(
                    np.sin(symmetry_order * angle_delta),
                    np.cos(symmetry_order * angle_delta),
                ) / symmetry_order
            ))

            # Clipping and the physics step can change the actual pose. Reject
            # states outside the requested limits or already successful in PushT.
            tolerance = 1e-8
            within_limits = (
                self.min_distance - tolerance <= actual_distance <= self.max_distance + tolerance
                and self.min_angle - tolerance <= actual_angle_error <= self.max_angle + tolerance
            )
            wrapper_success = actual_distance < SUCCESS_DISTANCE and actual_angle_error < SUCCESS_ANGLE
            if within_limits and not wrapper_success and base._get_coverage() <= base.success_threshold:
                break
        else:
            raise RuntimeError(f"Could not sample a valid curriculum state in {self.max_attempts} attempts")

        info["initial_goal_distance"] = float(actual_distance)
        info["initial_angle_error"] = float(actual_angle_error)
        metrics = self._metrics(info)
        self._previous = metrics
        self._was_contact = metrics["contact"]
        self._contact_events_this_episode = 0

        return obs, {**info, **metrics}

    def _potential(self, metrics):
        base = cast(PushTEnv, self.unwrapped)
        angle_error_scale = np.pi / getattr(base, "rotation_symmetry_order", 1)
        return -(
            self.agent_object_progress_weight * metrics["agent_object_distance"] / self.distance_scale
            + self.object_goal_progress_weight * metrics["object_goal_distance"] / self.distance_scale
            + self.angle_progress_weight * metrics["object_goal_angle_error"] / angle_error_scale
        )

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)

        # Override info["pos_agent"] with the true physical position from pymunk
        # because the default tracks the commanded target (red crosshair).
        base = cast(PushTEnv, self.unwrapped)
        info["pos_agent"] = np.array(base.agent.position, dtype=np.float32)

        metrics = self._metrics(info)

        # to incentivize contact initially in learning, reward based off distance to object
        dist_reward = (1.0 - metrics["agent_object_distance"] / self.distance_scale)

        # INVERTED: closer distance to goal gives higher reward (1.0 is right on the goal)
        block_goal_dist = 1.0 - (metrics["object_goal_distance"] / self.distance_scale)

        # INVERTED: lower angle error gives higher reward (1.0 is perfectly aligned)
        base = cast(PushTEnv, self.unwrapped)
        angle_error_scale = np.pi / getattr(base, "rotation_symmetry_order", 1)
        angle_reward = 1.0 - (metrics["object_goal_angle_error"] / angle_error_scale)

        # negative reward for going near bounds
        bound_reward = 0

        # adjust termination condition and bonus reward
        episode_end_reward = 0
        if metrics["object_goal_distance"] < SUCCESS_DISTANCE and metrics["object_goal_angle_error"] < SUCCESS_ANGLE:
            # within 15 pixels and 10 degrees
            terminated = True
            episode_end_reward = 500  # Massive bonus to encourage finishing the task

        # Decay the bonus for each new contact event within this episode.
        new_contact = metrics["contact"] and not self._was_contact
        contact_reward = (
            self.contact_bonus * self.contact_reward_decay ** self._contact_events_this_episode
            if new_contact else 0.0
        )
        if new_contact:
            self._contact_events_this_episode += 1

        step_penalty = -4.0  # Increased penalty to offset dense reward milking

        previous_potential = self._potential(self._previous)
        next_potential = 0.0 if terminated else self._potential(metrics)
        progress_reward = self.discount_gamma * next_potential - previous_potential

        reward = (
            self.agent_object_weight * dist_reward
            + block_goal_dist * 0.3
            + angle_reward * 0.07
            + episode_end_reward * 0.1
            + step_penalty * 0.1
            + bound_reward
            + contact_reward
            + progress_reward
        )

        info = {
            **info,
            **metrics,
            "dist_reward": float(dist_reward),
            "agent_object_weight": float(self.agent_object_weight),
            "progress_reward": float(progress_reward),
            "block_goal_dist": float(block_goal_dist),
            "angle_reward": float(angle_reward),
            "episode_end_reward": float(episode_end_reward),
            "step_penalty": float(step_penalty),
            "contact_reward": float(contact_reward),
            "contact_events_this_episode": self._contact_events_this_episode,
        }

        self._previous = metrics
        self._was_contact = metrics["contact"]
        return observation, reward, terminated, truncated, info

class ActionSmoothnessPenaltyWrapper(gym.Wrapper):
    """Penalizes large changes in action to encourage smooth continuous behavior."""
    def __init__(self, env, penalty_weight=0.1):
        super().__init__(env)
        self.penalty_weight = penalty_weight
        self.last_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Vector distance penalizes changes in both direction and magnitude.
        if self.last_action is not None:
            jitter_penalty = -self.penalty_weight * np.linalg.norm(action - self.last_action)
        else:
            jitter_penalty = 0.0

        self.last_action = action.copy()

        # Add penalty to the reward
        reward = float(reward) + jitter_penalty
        info['jitter_penalty'] = float(jitter_penalty)

        return obs, reward, terminated, truncated, info
