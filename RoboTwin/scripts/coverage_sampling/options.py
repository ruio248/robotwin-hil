"""Opt-in CLI; importing the normal evaluator does not load Torch/vision."""
from .core import SamplingConfig


def add_arguments(parser):
    parser.add_argument("--es-mode", choices=("off", "vanilla", "enhanced", "gaussian"), default="off")
    parser.add_argument("--es-window-start", type=int, help="First active decision index, zero-based")
    parser.add_argument("--es-window-end", type=int, help="Last active decision index, inclusive")
    parser.add_argument("--es-num-candidates", type=int, default=4)
    parser.add_argument("--es-policy-microbatch", type=int, default=4,
                        help="Candidate chunks per request (2..32); current 4090 JAX 0.30 profile supports <=8")
    parser.add_argument("--es-horizon", type=int, default=10, help="Expected FULL policy chunk length")
    parser.add_argument("--es-beta", type=float, default=None,
                        help="Legacy TD defaults to 10; discriminator uses frozen calibration unless overridden")
    parser.add_argument("--es-scorer", choices=("td_first_action", "discriminator_chunk"), default="td_first_action")
    parser.add_argument("--es-calibration", help="Completed benchmark fit/calibration.json (required for discriminator)")
    parser.add_argument("--es-calibration-key", default="disc_s42")
    parser.add_argument("--es-seed", type=int, default=42)
    parser.add_argument("--es-replay-atol", type=float, default=1e-4)
    parser.add_argument("--es-low-threshold", type=float, default=None)
    parser.add_argument(
        "--es-gaussian-sigma", type=float, default=None,
        help=("Gaussian baseline standard deviation in radians for every arm-joint entry of the "
              "complete output chunk; grippers are unchanged; zero is the uniform-mean control"),
    )
    parser.add_argument("--es-critic", help="Trusted full or inference-only tail critic checkpoint")
    parser.add_argument("--es-encoder-weights", help="Local ImageNet ResNet18 IMAGENET1K_V1 weights")
    parser.add_argument("--es-device", default="cpu")
    parser.add_argument("--es-log-dir", default=None)


def config_from_args(args):
    mode = args.get("es_mode", "off")
    if mode == "off":
        return None
    if str(args.get("eval_batch", False)).lower() in {"true", "1"}:
        raise ValueError("Enhanced sampling v1 supports serial eval only")
    if args.get("task_name") != "handover_to_tray" or args.get("action_type", "joint") not in {"joint", "qpos"}:
        raise ValueError("Enhanced sampling v1 supports handover_to_tray absolute joints only")
    if args.get("es_window_start") is None or args.get("es_window_end") is None:
        raise ValueError("Specify the same --es-window-start/end in both experiment arms")
    common = dict(
        mode=mode, window_start=args["es_window_start"], window_end=args["es_window_end"],
        num_candidates=args.get("es_num_candidates", 4),
        policy_microbatch=args.get("es_policy_microbatch", 4),
        horizon=args.get("es_horizon", 10), seed=args.get("es_seed", 42),
        replay_atol=args.get("es_replay_atol", 1e-4),
        low_threshold=args.get("es_low_threshold"),
    )
    if mode == "gaussian":
        if args.get("es_calibration") or args.get("es_beta") is not None:
            raise ValueError("Gaussian mode does not use coverage calibration or beta")
        return SamplingConfig(
            **common, beta=0.0, uniform_weights=True,
            gaussian_sigma=args.get("es_gaussian_sigma"),
        )
    if args.get("es_gaussian_sigma") is not None:
        raise ValueError("--es-gaussian-sigma is only valid with --es-mode gaussian")
    if not args.get("es_critic") or not args.get("es_encoder_weights"):
        raise ValueError("Both --es-critic and --es-encoder-weights are required")
    scorer = args.get("es_scorer", "td_first_action")
    beta = args.get("es_beta")
    extra = {}
    if scorer == "discriminator_chunk":
        from .calibration import load_discriminator_calibration
        if not args.get("es_calibration"):
            raise ValueError("Chunk discriminator requires --es-calibration")
        key = args.get("es_calibration_key", "disc_s42")
        calibration, digest = load_discriminator_calibration(args["es_calibration"], key, args["es_critic"])
        beta = calibration["beta"] if beta is None else beta
        extra = dict(score_representation="full_chunk", score_scale=calibration["scale"],
                     uniform_weights=calibration["degenerate"], calibration_sha256=digest, calibration_key=key)
    elif scorer != "td_first_action":
        raise ValueError("Unknown --es-scorer")
    elif args.get("es_calibration"):
        raise ValueError("Do not apply discriminator calibration to a legacy TD scorer")
    return SamplingConfig(**common, beta=10.0 if beta is None else beta, **extra)
