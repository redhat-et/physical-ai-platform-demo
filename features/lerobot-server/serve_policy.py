import argparse
import logging
import socket
import time

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.policies.factory import make_pre_post_processors

from websocket_policy_server import WebsocketPolicyServer


logger = logging.getLogger(__name__)


class LeRobotPolicyAdapter:
    """
    Adapter that exposes a LeRobot policy using the OpenPI BasePolicy-style
    infer() interface expected by WebsocketPolicyServer.
    """

    def __init__(
        self,
        policy,
        preprocessor,
        postprocessor,
    ):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.metadata = {
            "backend": "lerobot",
            "policy_type": policy.config.type,
            "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps,
        }

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()

        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    @staticmethod
    def _convert_observation(obs: dict) -> dict:
        """
        Convert playground/OpenPI-style observation into the feature names
        stored in this LeRobot checkpoint.
        """

        result = {}

        #
        # Images
        #
    
        if "observation.images.image" in obs:
            result["observation.images.image"] = obs[
                "observation.images.image"
            ]
    
        if "observation.images.image2" in obs:
            result["observation.images.image2"] = obs[
                "observation.images.image2"
            ]
    
        #
        # Pi0.5 checkpoint was trained with one empty camera slot.
        #
        if "observation.images.empty_camera_0" in obs:
            result["observation.images.empty_camera_0"] = obs[
                "observation.images.empty_camera_0"
            ]
        else:
            result["observation.images.empty_camera_0"] = np.zeros(
                (224, 224, 3),
                dtype=np.uint8,
            )
    
        #
        # State
        #
    
        if "observation.state" in obs:
            result["observation.state"] = obs[
                "observation.state"
            ]
    
        #
        # Prompt / task
        #
    
        if "task" in obs:
            result["task"] = obs["task"]
        elif "prompt" in obs:
            result["task"] = obs["prompt"]
        else:
            result["task"] = ""
    
        return result

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        observation = self._convert_observation(obs)

        #
        # 1. Native checkpoint preprocessor
        #
        batch = self.preprocessor(observation)

        #
        # 2. Native LeRobot Pi05 inference
        #
        actions = self.policy.predict_action_chunk(batch)

        if actions.ndim == 2:
            actions = actions.unsqueeze(0)

        #
        # actions:
        # (B, chunk_size, action_dim)
        #

        actions = actions[:, : self.policy.config.n_action_steps, :]

        #
        # 3. Native checkpoint postprocessor
        #
        # Current LeRobot async inference processes actions one timestep
        # at a time because the postprocessor expects (B, action_dim).
        #
        processed_actions = []

        for i in range(actions.shape[1]):
            action = actions[:, i, :]
            action = self.postprocessor(action)
            processed_actions.append(action)

        actions = torch.stack(processed_actions, dim=1)

        #
        # Remove batch dimension:
        # (1, T, 7) -> (T, 7)
        #
        actions = actions.squeeze(0)

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        return {
            "actions": actions,
        }


def create_policy(checkpoint: str, device: str):
    logger.info("Loading LeRobot checkpoint from %s", checkpoint)

    #
    # Load config.json FROM THE CHECKPOINT.
    #
    config = PreTrainedConfig.from_pretrained(
        checkpoint,
    )

    logger.info("Policy type: %s", config.type)
    logger.info("Chunk size: %s", config.chunk_size)
    logger.info("Action steps: %s", config.n_action_steps)

    #
    # Override runtime device.
    #
    config.device = device

    #
    # Get correct policy implementation based on config.type.
    #
    policy_class = get_policy_class(config.type)

    #
    # Load model.safetensors using checkpoint config.
    #
    policy = policy_class.from_pretrained(
        checkpoint,
        config=config,
    )

    policy = policy.to(device)
    policy.eval()

    #
    # IMPORTANT:
    #
    # Load policy_preprocessor.json,
    # policy_postprocessor.json,
    # and their safetensor state files FROM CHECKPOINT.
    #
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {
                "device": device,
            },
        },
        postprocessor_overrides={
            "device_processor": {
                "device": "cpu",
            },
        },
    )

    return LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
    )


def warmup(policy):
    logger.info("Running LeRobot warmup inference...")

    obs = {
        "observation.images.image": np.zeros(
            (256, 256, 3),
            dtype=np.uint8,
        ),
        "observation.images.image2": np.zeros(
            (256, 256, 3),
            dtype=np.uint8,
        ),
        "observation.images.empty_camera_0": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
        "observation.state": np.zeros(
            8,
            dtype=np.float32,
        ),
        "prompt": "warmup",
    }    
    
    t0 = time.monotonic()

    result = policy.infer(obs)

    logger.info(
        "Warmup completed in %.1fs, output shape=%s",
        time.monotonic() - t0,
        result["actions"].shape,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="/mnt/models",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )

    args = parser.parse_args()

    policy = create_policy(
        checkpoint=args.checkpoint,
        device=args.device,
    )

    if not args.skip_warmup:
        warmup(policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    logger.info(
        "Starting LeRobot policy server host=%s ip=%s port=%d",
        hostname,
        local_ip,
        args.port,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )

    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
    )

    main()
