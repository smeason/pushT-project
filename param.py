"""Parameters"""
class params():
    """PPO parameters"""
    policy_type: str = "mlp"
    state_dim: int = 5 # 5 base state features * 4 stacked frames
    action_dim: int = 2
    action_std: float = 0.3
    action_std_decay_rate: float = 1
    min_action_std: float = 0.05

    lr_actor: float = 0.0001
    lr_critic: float = 0.0003
    max_ppo_steps: int = 300
    max_episode_len: int = 400
    clip: float = 0.2
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 7
    batch_size: int = 32
    num_envs: int = 4
    vector_env_mode: str = "async"  # "sync" avoids per-worker Python/PyTorch memory overhead
    rollout_steps: int = 2048  # policy decisions per environment per update
    seed: int = 7689
    block_shape: str = "rectangle"  # "t", "square", "rectangle", or "polygon"
    block_width: float = 50.0
    block_height: float = 80.0
    block_vertices = None  # convex polygon vertices in local (x, y) coordinates
    rotation_symmetry_order = None  # infer for square/rectangle; default 1 for polygon
    agent_friction = None  # None keeps the Pymunk shape default
    block_friction = None
    wall_friction = None
    environment_damping = None  # None keeps PushT's current space damping
    entropy_coefficient: float = 0.010 # 0.01 default # 0.013 is way too high
    entropy_coefficient_min: float = 0.008
    smoothness_penalty_weight: float = 0.02
    initial_agent_object_weight: float = 0.081
    final_agent_object_weight: float = 0.08
    agent_object_progress_weight: float = 1.1 # 1.0
    object_goal_progress_weight: float = 1.1 # 1.0
    angle_progress_weight: float = 0.3 # 0.25
    contact_reward_decay: float = 0.5

    initial_min_dist: int = 20
    initial_min_angle: int = 15 # degrees
    initial_max_dist: int = 40 # Reduced from 90 to make initial learning possible
    initial_max_angle = 25 # Reduced from 120 degrees
    max_speed = 130 # maximum speed of pusher arm
    min_speed = 25
    action_every_n_steps = 4
    frame_stack_size = 8

    def print_params(self):
        ############# print all hyperparameters #############

        print("--------------------------------------------------------------------------------------------")

        print("max training timesteps : ", self.max_ppo_steps)
        print("max timesteps per episode : ", self.max_episode_len)

        # print("model saving frequency : " + str(save_model_freq) + " timesteps")
        # print("log frequency : " + str(log_freq) + " timesteps")
        # print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

        print("--------------------------------------------------------------------------------------------")

        print("state space dimension : ", self.state_dim)
        print("action space dimension : ", self.action_dim)

        print("--------------------------------------------------------------------------------------------")

        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", self.action_std)
        print("decay rate of std of action distribution : ", self.action_std_decay_rate)
        print("minimum std of action distribution : ", self.min_action_std)
        # print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")

        print("--------------------------------------------------------------------------------------------")

        # print("PPO update frequency : " + str() + " timesteps")
        print("PPO K epochs : ", self.update_epochs)
        print("PPO epsilon clip : ", self.clip)
        print("discount factor (gamma) : ", self.discount_gamma)

        print("--------------------------------------------------------------------------------------------")

        print("optimizer learning rate actor : ", self.lr_actor)
        print("optimizer learning rate critic : ", self.lr_critic)
