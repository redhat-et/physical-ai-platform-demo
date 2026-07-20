"""OpenPI policy server wrapper that runs a warmup inference before accepting connections.

Prevents Triton kernel autotuning from happening on the first real client request,
which can take 5+ minutes with pytorch_compile_mode='max-autotune'.
"""

import logging
import socket
import time
from dataclasses import dataclass, field

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training import config as _config

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    config: str
    dir: str


@dataclass
class Default:
    pass


@dataclass
class Args:
    policy: Checkpoint | Default = field(default_factory=Default)
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False


WARMUP_SCHEMAS: dict[str, dict] = {
    "droid": {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "warmup",
    },
    "aloha": {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_low": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": "warmup",
    },
    "libero": {
        "observation/state": np.zeros(8, dtype=np.float64),
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "prompt": "warmup",
    },
}

DEFAULT_FAMILY = "droid"


def _detect_config_family(config_name: str) -> str:
    """Extract the robot family from a config name like 'pi05_droid' or 'pi0_aloha_sim'."""
    config_lower = config_name.lower()
    for family in WARMUP_SCHEMAS:
        if family in config_lower:
            return family
    logger.warning(
        "Unknown config family for '%s', falling back to '%s' warmup schema",
        config_name,
        DEFAULT_FAMILY,
    )
    return DEFAULT_FAMILY


def _warmup(policy, config_name: str) -> None:
    """Run a single dummy inference to trigger Triton kernel compilation."""
    family = _detect_config_family(config_name)
    obs = WARMUP_SCHEMAS[family]
    logger.info("Running warmup inference (config family: %s)...", family)
    t0 = time.monotonic()
    policy.infer(obs)
    logger.info("Warmup inference completed in %.1fs", time.monotonic() - t0)


def create_policy(args: Args):
    match args.policy:
        case Checkpoint(config=config, dir=dir):
            train_config = _config.get_config(config)
            return _policy_config.create_trained_policy(train_config, dir), config
        case Default():
            raise ValueError("Warmup wrapper requires explicit --policy.config")


def main(args: Args) -> None:
    policy, config_name = create_policy(args)

    if args.record:
        from openpi.serving import policy_recorder

        policy = policy_recorder.PolicyRecorder(policy)

    _warmup(policy, config_name)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
