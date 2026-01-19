from src.core.rl_modules.impala_rl_modules import ImpalaMlpModule
from ray.rllib.core.rl_module import RLModuleSpec


def add_cli_args(parser):
    """Register only the flags IMPALA cares about."""
    parser.add_argument("--env-type", type=str, default="discrete",
                        help="whether environment action space is 'continuous' or 'discrete'.")
    return parser


def update_config(cfg, args):
    """Add/override IMPALA-specific settings on the RLlib config builder."""

    # wrap custom RLModule in the module spec
    spec = RLModuleSpec(
        module_class=ImpalaMlpModule,
        observation_space=cfg.observation_space,
        action_space=cfg.action_space,
    )
    return (
        cfg.training(
            vtrace_clip_rho_threshold=1.0,
            vtrace_clip_pg_rho_threshold=1.0,
        )
        .rl_module(rl_module_spec=spec)
        # .rl_module(model_config={"vf_share_layers": False})
    )
