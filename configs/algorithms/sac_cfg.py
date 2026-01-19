def add_cli_args(parser):
    """Register only the flags SAC cares about."""
    parser.add_argument("--env-type", type=str, default="continuous",
                        help="whether environment action space is 'continuous' or 'discrete'.")
    return parser


def update_config(cfg, args):
    """Add/override APPO-specific settings on the RLlib config builder."""
    return (
        cfg.training(
            model={"free_log_std": True},
            q_model_config={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define custom Q-model(s).
                "custom_model_config": {},
            },
            policy_model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "post_fcnet_hiddens": [],
                "post_fcnet_activation": None,
                "custom_model": None,  # Use this to define a custom policy model.
                "custom_model_config": {},
            },
            initial_alpha=0.1,
            alpha_lr=1e-4,
            actor_lr=1e-4,
            critic_lr=1e-4,
            tau=0.003,
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                # Size of the replay buffer. Note that if async_updates is set,
                # then each worker will have a replay buffer of this size.
                "capacity": int(5e4),
                "alpha": 0.6,
                # Beta parameter for sampling from prioritized replay buffer.
                "beta": 0.4,
            }
        )
        # you can chain more .XYZ() builders here
    )
