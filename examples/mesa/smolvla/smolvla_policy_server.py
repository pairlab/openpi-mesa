"""SmolVLA policy server for mesa bimanual eval.

Loads a trained SmolVLA lerobot checkpoint and serves it over the openpi
msgpack-numpy websocket protocol so `run-vla-benchmark.sh` can drive it.

Wire contract (matches eval_policy_server.py):
    obs["images"][cam_name] : (H, W, 3) uint8
        Expected cameras: egocentric, robot0_eye_in_hand, robot1_eye_in_hand
    obs["state"]            : (14,) float32  [jp0(6), g0(1), jp1(6), g1(1)]
    obs["prompt"]           : str

Returns:
    {"actions": (chunk_size, 14) float32} absolute joint-position targets
"""

import dataclasses
import logging
import socket

import numpy as np
import torch
import tyro

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from openpi.serving import websocket_policy_server


@dataclasses.dataclass
class Args:
    # Path to the lerobot pretrained_model dir (e.g. .../checkpoints/000050000/pretrained_model)
    checkpoint_dir: str

    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str | None = None


class SmolVLAMesaBimanualPolicy:
    """Wraps SmolVLA for the mesa bimanual eval wire format."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda", default_prompt: str | None = None):
        self._device = device
        self._default_prompt = default_prompt

        policy_cls = get_policy_class("smolvla")
        self._policy = policy_cls.from_pretrained(checkpoint_dir)
        self._policy.to(device)
        self._policy.eval()
        torch.set_grad_enabled(False)

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            checkpoint_dir,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        self.metadata = {
            "model": "SmolVLA",
            "action_dim": self._policy.config.output_features["action"].shape[0],
            "action_horizon": self._policy.config.chunk_size,
            "hostname": socket.gethostname(),
        }

    def infer(self, obs: dict) -> dict:
        prompt = obs.get("prompt") or self._default_prompt or ""
        images = obs["images"]

        lerobot_obs = {}

        for cam in ("egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand"):
            img = np.asarray(images[cam])
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
            lerobot_obs[f"observation.images.{cam}"] = (
                torch.from_numpy(img).unsqueeze(0).to(self._device)
            )

        state = np.asarray(obs["state"], dtype=np.float32)
        lerobot_obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(self._device)
        lerobot_obs["task"] = prompt

        lerobot_obs = self._preprocess(lerobot_obs)

        with torch.no_grad():
            actions = self._policy.predict_action_chunk(lerobot_obs)

        actions = self._postprocess(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        return {"actions": actions}

    def reset(self) -> None:
        self._policy.reset()


def main(args: Args) -> None:
    policy = SmolVLAMesaBimanualPolicy(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        default_prompt=args.default_prompt,
    )
    logging.info(
        "Starting SmolVLA mesa bimanual eval server host=%s port=%d ckpt=%s",
        args.host, args.port, args.checkpoint_dir,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
