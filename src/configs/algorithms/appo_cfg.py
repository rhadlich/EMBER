from core.rl_modules.appo_rl_modules import AppoMlpModule
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


def add_cli_args(parser):
    """Register only the flags APPO cares about."""
    parser.add_argument("--env-type", type=str, default="continuous",
                        help="whether environment action space is 'continuous' or 'discrete'.")
    return parser


def update_config(cfg, args):
    """Add/override APPO-specific settings on the RLlib config builder."""
    # # wrap custom RLModule in the module spec
    # spec = RLModuleSpec(
    #     module_class=AppoMlpModule,
    #     observation_space=cfg.observation_space,
    #     action_space=cfg.action_space,
    # )
    return (
        cfg.training(
            lr=3e-5,  # one optimiser for both heads
            vf_loss_coeff=0.5,  # value-loss weight
            entropy_coeff=0.01,  # policy entropy bonus
            grad_clip=40.0,
            tau=0.005,  # target-net polyak factor (APPO-only)
            # other APPO-specific knobs:
            vtrace=True,
            clip_param=0.4,
            target_network_update_freq=2,
        )
        .rl_module(
            model_config=DefaultModelConfig(
                # shared encoder (actor+vf) analogous to your policy net
                fcnet_hiddens=[128, 128],
                fcnet_activation="relu",
                # value head only; policy head is implicit
                head_fcnet_hiddens=[64, 64],
                vf_share_layers=False,  # False → separate value head
                free_log_std=True  # same as SAC’s ``model["free_log_std"]``
            )
        )
        # .rl_module(rl_module_spec=spec)
    )
