import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MESAInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MESABimanualInputs(transforms.DataTransformFn):
    """Three-camera bimanual variant of :class:`MESAInputs`.

    Expects ``observation/wrist_image_right`` for the second wrist view instead
    of the zero-filled placeholder used by the single-arm class.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist_image = _parse_image(data["observation/wrist_image_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MESABimanualAdapt3RInputs(transforms.DataTransformFn):
    """Bimanual Mesa inputs for the 3D-positional Pi0Adapt3R model.

    Extends :class:`MESABimanualInputs` with per-camera depth + calibration (intrinsics,
    extrinsics) and the reference-arm pose pair used to transform the point cloud into
    the end-effector frame. ``hand_mat_inv`` is computed here to keep the LeRobot dataset
    footprint minimal.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = MESABimanualInputs(model_type=self.model_type)(data)

        inputs["depth"] = {
            "base_0_depth": _parse_depth(data["observation/depth_base"]),
            "left_wrist_0_depth": _parse_depth(data["observation/depth_left_wrist"]),
            "right_wrist_0_depth": _parse_depth(data["observation/depth_right_wrist"]),
        }

        hand_mat = np.asarray(data["observation/hand_mat_robot0"], dtype=np.float32)
        inputs["calibration"] = {
            "base_intrinsics": np.asarray(data["observation/intrinsics_base"], dtype=np.float32),
            "base_extrinsics": np.asarray(data["observation/extrinsics_base"], dtype=np.float32),
            "left_wrist_intrinsics": np.asarray(data["observation/intrinsics_left_wrist"], dtype=np.float32),
            "left_wrist_extrinsics": np.asarray(data["observation/extrinsics_left_wrist"], dtype=np.float32),
            "right_wrist_intrinsics": np.asarray(data["observation/intrinsics_right_wrist"], dtype=np.float32),
            "right_wrist_extrinsics": np.asarray(data["observation/extrinsics_right_wrist"], dtype=np.float32),
            "hand_mat": hand_mat,
            "hand_mat_inv": np.linalg.inv(hand_mat).astype(np.float32),
        }
        return inputs


@dataclasses.dataclass(frozen=True)
class MESABimanualAdapt3RMultiCamInputs(transforms.DataTransformFn):
    """N-camera bimanual Mesa inputs for Pi0Adapt3R.

    Generalizes :class:`MESABimanualAdapt3RInputs` to arbitrary camera sets. Each entry in
    ``cameras`` names one camera whose LeRobot features have been repacked to keys of the
    form ``observation/{image,depth,intrinsics,extrinsics}_{cam}``; the transform emits
    model-side entries ``{cam}_0_rgb``, ``{cam}_0_depth``, ``{cam}_intrinsics``,
    ``{cam}_extrinsics``. The reference arm for ``hand_mat`` is selectable via ``ref_arm``.
    """

    model_type: _model.ModelType
    cameras: tuple[str, ...]
    ref_arm: int = 0

    def __call__(self, data: dict) -> dict:
        if not self.cameras:
            raise ValueError("MESABimanualAdapt3RMultiCamInputs.cameras must be non-empty.")

        image_dict: dict[str, np.ndarray] = {}
        mask_dict: dict[str, np.bool_] = {}
        depth_dict: dict[str, np.ndarray] = {}
        calibration: dict[str, np.ndarray] = {}

        for cam in self.cameras:
            image_dict[f"{cam}_0_rgb"] = _parse_image(data[f"observation/image_{cam}"])
            mask_dict[f"{cam}_0_rgb"] = np.True_
            depth_dict[f"{cam}_0_depth"] = _parse_depth(data[f"observation/depth_{cam}"])
            calibration[f"{cam}_intrinsics"] = np.asarray(
                data[f"observation/intrinsics_{cam}"], dtype=np.float32
            )
            calibration[f"{cam}_extrinsics"] = np.asarray(
                data[f"observation/extrinsics_{cam}"], dtype=np.float32
            )

        hand_mat = np.asarray(data[f"observation/hand_mat_robot{self.ref_arm}"], dtype=np.float32)
        calibration["hand_mat"] = hand_mat
        calibration["hand_mat_inv"] = np.linalg.inv(hand_mat).astype(np.float32)

        inputs = {
            "state": data["observation/state"],
            "image": image_dict,
            "image_mask": mask_dict,
            "depth": depth_dict,
            "calibration": calibration,
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


def _parse_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize a stored depth array to ``(H, W, 1)`` float32 (undo LeRobot's CHW layout)."""
    depth = np.asarray(depth, dtype=np.float32)
    # Stored as (1, H, W); transpose to (H, W, 1) to match the RGB HWC convention used downstream.
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = np.transpose(depth, (1, 2, 0))
    elif depth.ndim == 2:
        depth = depth[..., None]
    return depth


@dataclasses.dataclass(frozen=True)
class MESAOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        result = {"actions": np.asarray(data["actions"][:, :self.action_dim])}
        return result
