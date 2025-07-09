def add_cli_args(parser):
    """Register only the flags APPO cares about."""
    parser.add_argument("--vtrace-rho", type=float, default=1.0,
                        help="ρ̄ clip for V-trace.")
    parser.add_argument("--vtrace-pg-rho", type=float, default=1.0,
                        help="ρ̄ clip for V-trace PG term.")
    parser.add_argument("--env-type", type=str, default="discrete",
                        help="whether environment action space is 'continuous' or 'discrete'.")
    return parser


def update_config(cfg, args):
    """Add/override APPO-specific settings on the RLlib config builder."""
    return (
        cfg.training(
            vtrace_clip_rho_threshold=args.vtrace_rho,
            vtrace_clip_pg_rho_threshold=args.vtrace_pg_rho,
        )
        # you can chain more .XYZ() builders here
    )
