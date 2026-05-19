"""Policy server for pi05 mesa bimanual overfit eval.

Loads a trained pi05 LoRA checkpoint and serves it over the openpi
msgpack-numpy websocket protocol so `vla-benchmark/scripts/eval_server_parallel.py`
can drive it unchanged.

Wire contract with the eval client (2D RGB-only):
    obs["images"][cam_name] : (H, W, 3) uint8
        3-cam configs: egocentric, robot0_eye_in_hand, robot1_eye_in_hand.
        Multi-cam (camdrop) configs: every camera listed in
        ``train_config.data.cameras`` (e.g. egocentric, leftshoulder,
        rightshoulder, midshoulder).
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
from openpi import transforms as _transforms


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

    # Comma-separated list of bare camera names whose ``image_mask`` stays True
    # at inference time; all other cameras have their mask zeroed (mirrors the
    # camdrop training-time semantics — image data is still sent, but the
    # transformer is told to ignore those tokens). Only honored for multi-cam
    # configs. Empty = use every camera (no masking).
    keep_cameras: str = ""


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


class MESABimanualMultiCamEvalWrapper:
    """Wire-translator for multi-cam bimanual configs (e.g. ``*_camdrop``).

    Iterates over the camera set declared on the train config rather than the
    fixed ``egocentric`` + 2 wrist views. ``include_depth=True`` adds the
    per-camera depth + intrinsics + extrinsics needed by the 3D positional head.
    """

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        cameras: tuple[str, ...],
        include_depth: bool,
        ref_arm: int = 0,
        extra_arms: tuple[int, ...] = (),
    ) -> None:
        if not cameras:
            raise ValueError("MESABimanualMultiCamEvalWrapper requires at least one camera.")
        self._policy = policy
        self._cameras = tuple(cameras)
        self._include_depth = include_depth
        self._ref_arm = ref_arm
        self._extra_arms = tuple(extra_arms)

    def infer(self, obs: dict) -> dict:
        logging.info(
            "infer: cams=%s state_shape=%s prompt=%r include_depth=%s",
            self._cameras,
            np.asarray(obs["state"]).shape,
            obs.get("prompt"),
            self._include_depth,
        )
        images = obs["images"]
        element: dict = {
            "observation/state": np.asarray(obs["state"], dtype=np.float32),
            "prompt": obs["prompt"],
        }
        for cam in self._cameras:
            element[f"observation/image_{cam}"] = images[cam]
        if self._include_depth:
            depth = obs["depth"]
            intrinsics = obs["intrinsics"]
            extrinsics = obs["extrinsics"]
            for cam in self._cameras:
                element[f"observation/depth_{cam}"] = depth[cam]
                element[f"observation/intrinsics_{cam}"] = intrinsics[cam]
                element[f"observation/extrinsics_{cam}"] = extrinsics[cam]
            element[f"observation/hand_mat_robot{self._ref_arm}"] = obs["hand_mat"][f"robot{self._ref_arm}"]
            for arm in self._extra_arms:
                element[f"observation/hand_mat_robot{arm}"] = obs["hand_mat"][f"robot{arm}"]
        result = self._policy.infer(element)
        logging.info("infer: actions shape=%s", np.asarray(result["actions"]).shape)
        return result

    def reset(self) -> None:
        self._policy.reset()


@dataclasses.dataclass(frozen=True)
class _ApplyImageMask(_transforms.DataTransformFn):
    """Zero ``image_mask`` entries whose key is not in ``keep_keys``.

    Inserted after the data-transform that builds the masks (e.g.
    :class:`MESABimanualAdapt3RMultiCamInputs`) so we can simulate the camdrop
    training-time ``image_mask=False`` for the non-egocentric cameras.
    ``keep_keys`` are the model-side mask keys (e.g. ``egocentric_0_rgb``).
    """

    keep_keys: frozenset[str]

    def __call__(self, data: dict) -> dict:
        masks = data["image_mask"]
        unknown = self.keep_keys - set(masks.keys())
        if unknown:
            raise ValueError(
                f"keep_cameras references unknown image_mask keys {sorted(unknown)}; "
                f"available: {sorted(masks.keys())}"
            )
        new_masks = {k: (v if k in self.keep_keys else np.False_) for k, v in masks.items()}
        return {**data, "image_mask": new_masks}


def _install_camera_mask(policy: _policy.Policy, keep_cameras: tuple[str, ...]) -> None:
    """Mutate ``policy`` so ``_input_transform`` zeros image_masks for cams not
    in ``keep_cameras``. No-op when ``keep_cameras`` is empty."""
    if not keep_cameras:
        return
    keep_keys = frozenset(f"{c}_0_rgb" for c in keep_cameras)
    base = policy._input_transform  # noqa: SLF001 — internal extension hook
    policy._input_transform = _transforms.compose([base, _ApplyImageMask(keep_keys=keep_keys)])  # noqa: SLF001
    logging.info("camera mask installed: keep=%s", sorted(keep_keys))


def _resolve_multi_cam(
    train_config: _config.TrainConfig,
) -> tuple[bool, tuple[str, ...], bool, tuple[int, ...]]:
    """Return (is_multi_cam, bare_cam_names, include_depth, extra_arms) for ``train_config``.

    Multi-cam is detected by the presence of the ``cameras`` attribute on the
    data config factory (set by :class:`MESABimanualAdapt3RMultiCamDataConfig`).
    """
    data_cfg = train_config.data
    cameras = getattr(data_cfg, "cameras", None)
    if cameras is None:
        return False, (), False, ()
    include_depth = bool(getattr(data_cfg, "include_depth", True))
    extra_arms = tuple(getattr(data_cfg, "extra_arms", ()))
    return True, tuple(cameras), include_depth, extra_arms


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    is_multi_cam, cameras, include_depth, extra_arms = _resolve_multi_cam(train_config)
    is_3d_legacy = (not is_multi_cam) and train_config.model.model_type == _model.ModelType.PI0_ADAPT3R
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata

    keep_cameras = tuple(c.strip() for c in args.keep_cameras.split(",") if c.strip())
    if keep_cameras and not is_multi_cam:
        raise ValueError(
            f"--keep-cameras is only supported for multi-cam configs; "
            f"{args.config!r} expects 3-cam wrapper."
        )
    if is_multi_cam:
        unknown = set(keep_cameras) - set(cameras)
        if unknown:
            raise ValueError(
                f"--keep-cameras references cams {sorted(unknown)} not in train cams {sorted(cameras)}"
            )

    _install_camera_mask(policy, keep_cameras)

    if is_multi_cam:
        wrapped = MESABimanualMultiCamEvalWrapper(
            policy,
            cameras=cameras,
            include_depth=include_depth,
            extra_arms=extra_arms,
        )
    else:
        wrapped = MESABimanualEvalWrapper(policy, is_3d=is_3d_legacy)

    hostname = socket.gethostname()
    logging.info(
        "Starting mesa bimanual eval policy server (host=%s, port=%d, config=%s, ckpt=%s, "
        "multi_cam=%s cams=%s include_depth=%s extra_arms=%s keep_cameras=%s)",
        hostname, args.port, args.config, args.checkpoint_dir,
        is_multi_cam, cameras, include_depth, extra_arms, keep_cameras,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
