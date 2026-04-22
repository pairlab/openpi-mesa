"""Policy server for pi05 mesa bimanual overfit eval.

Loads a trained pi05 LoRA checkpoint and serves it over the openpi
msgpack-numpy websocket protocol so `vla-benchmark/scripts/eval_server_parallel.py`
can drive it unchanged.

Wire contract with the eval client (2D RGB-only):
    obs["images"][cam_name] : (H, W, 3) uint8
        Expected cameras (in any order): egocentric, robot0_eye_in_hand,
        robot1_eye_in_hand.
    obs["state"]            : (14,) float32, packed in the order of
        --state-keys robot0_joint_pos robot0_gripper_jaw_width
                     robot1_joint_pos robot1_gripper_jaw_width
        i.e. [jp0(6), g0(1), jp1(6), g1(1)] — matches training layout.
    obs["prompt"]           : str

3D variants (Pi0Adapt3R, e.g. pi05_mesa_bimanual_lora_3d) additionally require:
    obs["depth"][cam_name]      : (H, W, 1) float32, meters.
    obs["intrinsics"][cam_name] : (3, 3) float32.
    obs["extrinsics"][cam_name] : (4, 4) float32, camera-to-world.
    obs["hand_mat"]["robot0"]   : (4, 4) float32, robot0 end-effector pose in world.

Returns:
    {"actions": (chunk_size, 14) array} = absolute joint-position targets
    [joint_pos0(6), grip0(1), joint_pos1(6), grip1(1)]; gripper in {-1, +1}
    (-1 = open, +1 = close).
"""

import dataclasses
import logging
import socket

import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Checkpoint step dir (e.g., .../pi05_bimanual_2task_lora_v2/9000).
    checkpoint_dir: str

    # Training config name used to reconstruct model + data transforms.
    config: str = "pi05_mesa_bimanual_lora"

    host: str = "0.0.0.0"
    port: int = 8000

    # Optional fallback prompt if the eval client does not provide one.
    default_prompt: str | None = None


class MESABimanualEvalWrapper:
    """Translate vla-benchmark wire obs into the pi05 bimanual repack format.

    Supports both 2D (RGB-only) and 3D (Pi0Adapt3R: RGB + depth + calibration
    + robot0 end-effector pose) variants; ``is_3d`` gates the extra keys.
    """

    def __init__(self, policy: _policy.Policy, *, is_3d: bool = False) -> None:
        self._policy = policy
        self._is_3d = is_3d

    def infer(self, obs: dict) -> dict:
        logging.info(
            "infer: image_keys=%s state_shape=%s state_dtype=%s prompt=%r is_3d=%s",
            list(obs.get("images", {}).keys()),
            np.asarray(obs["state"]).shape,
            np.asarray(obs["state"]).dtype,
            obs.get("prompt"),
            self._is_3d,
        )
        images = obs["images"]
        element = {
            "observation/image": images["egocentric"],
            "observation/wrist_image": images["robot0_eye_in_hand"],
            "observation/wrist_image_right": images["robot1_eye_in_hand"],
            "observation/state": np.asarray(obs["state"], dtype=np.float32),
            "prompt": obs["prompt"],
        }
        if self._is_3d:
            depth = obs["depth"]
            intrinsics = obs["intrinsics"]
            extrinsics = obs["extrinsics"]
            element.update({
                "observation/depth_base": depth["egocentric"],
                "observation/depth_left_wrist": depth["robot0_eye_in_hand"],
                "observation/depth_right_wrist": depth["robot1_eye_in_hand"],
                "observation/intrinsics_base": intrinsics["egocentric"],
                "observation/intrinsics_left_wrist": intrinsics["robot0_eye_in_hand"],
                "observation/intrinsics_right_wrist": intrinsics["robot1_eye_in_hand"],
                "observation/extrinsics_base": extrinsics["egocentric"],
                "observation/extrinsics_left_wrist": extrinsics["robot0_eye_in_hand"],
                "observation/extrinsics_right_wrist": extrinsics["robot1_eye_in_hand"],
                "observation/hand_mat_robot0": obs["hand_mat"]["robot0"],
            })
        logging.info("infer: calling underlying policy.infer")
        result = self._policy.infer(element)
        logging.info("infer: policy returned, actions shape=%s", np.asarray(result["actions"]).shape)
        return result

    def reset(self) -> None:
        self._policy.reset()


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    is_3d = train_config.model.model_type == _model.ModelType.PI0_ADAPT3R
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata
    policy = MESABimanualEvalWrapper(policy, is_3d=is_3d)

    hostname = socket.gethostname()
    logging.info(
        "Starting mesa bimanual eval policy server (host=%s, port=%d, config=%s, ckpt=%s, is_3d=%s)",
        hostname, args.port, args.config, args.checkpoint_dir, is_3d,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
