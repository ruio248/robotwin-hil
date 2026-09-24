"""Opt-in CLI; importing the normal evaluator does not load Torch/vision."""
from .core import SamplingConfig


def add_arguments(parser):
    parser.add_argument("--es-mode", choices=("off", "vanilla", "enhanced"), default="off")
    parser.add_argument("--es-window-start", type=int, help="First active decision index, zero-based")
    parser.add_argument("--es-window-end", type=int, help="Last active decision index, inclusive")
    parser.add_argument("--es-num-candidates", type=int, default=4)
    parser.add_argument("--es-horizon", type=int, default=10, help="Expected FULL policy chunk length")
    parser.add_argument("--es-beta", type=float, default=10.0)
    parser.add_argument("--es-seed", type=int, default=42)
    parser.add_argument("--es-replay-atol", type=float, default=1e-4)
    parser.add_argument("--es-low-threshold", type=float, default=None)
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
    if not args.get("es_critic") or not args.get("es_encoder_weights"):
        raise ValueError("Both --es-critic and --es-encoder-weights are required")
    if args.get("es_window_start") is None or args.get("es_window_end") is None:
        raise ValueError("Specify the same --es-window-start/end in both experiment arms")
    return SamplingConfig(mode=mode, window_start=args["es_window_start"], window_end=args["es_window_end"],
                          num_candidates=args.get("es_num_candidates", 4), horizon=args.get("es_horizon", 10),
                          beta=args.get("es_beta", 10.0), seed=args.get("es_seed", 42),
                          replay_atol=args.get("es_replay_atol", 1e-4), low_threshold=args.get("es_low_threshold"))
