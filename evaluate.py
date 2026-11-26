from pathlib import Path
import time

import numpy as np
import torch

from env_setup import make_env
from param import params
from ppo import PPO

def save_frames_as_gif(frames, path, fps=10, frame_stride=1, max_size=(512, 512), frame_rewards=None, frame_steps=None):
    from PIL import Image, ImageDraw

    if not frames:
        raise ValueError("Cannot save a GIF with no frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if frame_rewards is not None and len(frame_rewards) != len(frames):
        raise ValueError("frame_rewards must have one value per frame")
    if frame_steps is not None and len(frame_steps) != len(frames):
        raise ValueError("frame_steps must have one value per frame")

    selected_indices = range(0, len(frames), frame_stride)
    images = []
    for frame_index in selected_indices:
        frame = frames[frame_index]
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(f"Expected an RGB image, got shape {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        image = Image.fromarray(np.ascontiguousarray(frame[:, :, :3]), mode="RGB")
        if max_size is not None:
            image.thumbnail(max_size, Image.Resampling.BILINEAR)
        if frame_rewards is not None or frame_steps is not None:
            step = frame_steps[frame_index] if frame_steps is not None else frame_index
            reward = frame_rewards[frame_index] if frame_rewards is not None else None
            reward_text = "--" if reward is None else f"{reward:+.3f}"
            label = f"Step {step} | Reward: {reward_text}"
            draw = ImageDraw.Draw(image)
            text_bounds = draw.textbbox((6, 5), label)
            draw.rectangle(
                (2, 2, text_bounds[2] + 4, text_bounds[3] + 3),
                fill=(0, 0, 0),
            )
            draw.text((6, 5), label, fill=(255, 255, 255))
        images.append(image)

    # GIF timing is quantized to 10 ms; preserve elapsed time when frames are skipped.
    duration = max(10, round((1000 * frame_stride / fps) / 10) * 10)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

def main():
    hp = params()
    torch.set_num_threads(1)

    # Evaluation uses one environment and applies action repeats explicitly below.
    env = make_env(
        hp,
        seed=hp.seed,
        action_repeat=1,
        max_episode_steps=hp.max_episode_len,
        discount_gamma=hp.discount_gamma,
    )()
    try:
        model = PPO(
            env,
            hp=hp,
            policy_type=hp.policy_type,
            discrete_actions=True,
            state_dim=hp.state_dim,
            action_dim=hp.action_dim,
            frame_stack_size=hp.frame_stack_size,
            lr_actor=hp.lr_actor,
            lr_critic=hp.lr_critic,
            max_ppo_steps=hp.max_ppo_steps,
            max_episode_len=hp.max_episode_len,
            clip=hp.clip,
            discount_gamma=hp.discount_gamma,
            gae_lambda=hp.gae_lambda,
            update_epochs=hp.update_epochs,
            batch_size=hp.batch_size,
            entropy_coefficient=hp.entropy_coefficient,
            action_every_n_steps=hp.action_every_n_steps,
        )
        checkpoint = Path(__file__).resolve().parent.parent / "ppo_mlp_histlen8.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Could not find policy checkpoint: {checkpoint}")
        model.load(str(checkpoint))

        num_test_rollouts = 10
        for rollout in range(num_test_rollouts):
            observation, info = env.reset(seed=hp.seed + rollout)
            frames = [env.render()]
            frame_rewards = [None]
            frame_steps = [0]
            episode_steps = 0
            episode_reward = 0.0
            terminated = truncated = False
            start_time = time.perf_counter()

            while episode_steps < model.MAX_EPISODE_LEN and not (terminated or truncated):
                state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    # Select the most likely bin for each action dimension so
                    # each evaluation rollout is repeatable.
                    distributions = model.policy_old.get_distribution(state)
                    action = torch.stack(
                        [distribution.logits.argmax(dim=-1) for distribution in distributions],
                        dim=-1,
                    )
                    environment_action = model._to_environment_action(action)

                steps_to_run = min(
                    model.action_every_n_steps,
                    model.MAX_EPISODE_LEN - episode_steps,
                )
                for _ in range(steps_to_run):
                    observation, reward, terminated, truncated, info = env.step(environment_action)
                    episode_reward += float(reward)
                    episode_steps += 1
                    if episode_steps % 2 == 0 or terminated or truncated:
                        frames.append(env.render())
                        frame_rewards.append(float(reward))
                        frame_steps.append(episode_steps)
                    if terminated or truncated:
                        break

            elapsed = time.perf_counter() - start_time
            gif_path = Path(__file__).resolve().parent / f"rolloutG_{rollout}.gif"
            save_frames_as_gif(
                frames, gif_path, fps=10, frame_stride=1, max_size=(256, 256),
                frame_rewards=frame_rewards, frame_steps=frame_steps,
            )
            print(
                f"Rollout {rollout + 1}/{num_test_rollouts} | "
                f"Reward: {episode_reward:.3f} | Steps: {episode_steps} | "
                f"Success: {bool(info.get('episode_end_reward', 0) > 0 or info.get('is_success', False))} | "
                f"Time: {elapsed:.3f}s | GIF: {gif_path.name}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
