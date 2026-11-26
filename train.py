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
from env_setup import make_env
from env_setup import DistanceCurriculumAndRewardWrapper, ActionWrapper, PushTVelocityWrapper, ActionSmoothnessPenaltyWrapper, NormalizeObservation
from ppo import PPO, RolloutBuffer
from vector_ppo import VectorPPO

def main():
    hp = params()
    torch.set_num_threads(1)
    env_fns = [make_env(
        hp, seed=hp.seed + i, action_repeat=hp.action_every_n_steps,
        max_episode_steps=hp.max_episode_len, discount_gamma=hp.discount_gamma,
    ) for i in range(hp.num_envs)]
    vector_env_class = {
        "sync": gym.vector.SyncVectorEnv,
        "async": gym.vector.AsyncVectorEnv,
    }.get(hp.vector_env_mode)
    if vector_env_class is None:
        raise ValueError("vector_env_mode must be 'sync' or 'async'")
    env = vector_env_class(
        env_fns,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        print(f"Vectorized Env Observation Space: {env.observation_space}")
        print(f"Vectorized Env Action Space: {env.action_space}")
        model = VectorPPO(
            env, hp=hp, rollout_steps=hp.rollout_steps, seed=hp.seed,
            policy_type=hp.policy_type, discrete_actions=True,
            state_dim=hp.state_dim, action_dim=hp.action_dim,
            frame_stack_size=hp.frame_stack_size, lr_actor=hp.lr_actor,
            lr_critic=hp.lr_critic, max_ppo_steps=hp.max_ppo_steps,
            max_episode_len=hp.max_episode_len, clip=hp.clip,
            discount_gamma=hp.discount_gamma, gae_lambda=hp.gae_lambda,
            update_epochs=hp.update_epochs, batch_size=hp.batch_size,
            entropy_coefficient=hp.entropy_coefficient,
            action_every_n_steps=hp.action_every_n_steps,
        )
        # checkpoint = Path(__file__).resolve().parent / "ppo_mlp_histlen8.pt"
        # if not checkpoint.is_file():
        #     raise FileNotFoundError(f"Could not find policy checkpoint: {checkpoint}")
        # model.load(str(checkpoint))
        x, y = model.run()
        plt.plot(x, y)
        plt.title("rewards during training")
        plt.xlabel('iteration')
        plt.ylabel('reward')
        plt.savefig("rewards_nov25_1.png", dpi=300, bbox_inches="tight")

        model.save("ppo_mlp_histlen8.pt")
    finally:
        env.close()


if __name__ == "__main__": 
    main()