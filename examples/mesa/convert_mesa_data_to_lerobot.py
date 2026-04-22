"""Convert Mesa bimanual HDF5 data to the LeRobot v2.0 dataset format.

Each HDF5 holds one or more demos under ``data/demo_N``. An episode is written
per demo. Actions are the 14-D ``actions_joint_pos`` (absolute joint-space
target: ``[joint_pos(6), grip(1)]`` per arm). State mirrors that layout,
derived from ``robot{0,1}_joint_pos`` and ``robot{0,1}_gripper_jaw_width``.

When ``--include-3d`` is set, also emits per-camera metric depth (linearized
from MuJoCo's z-buffer), per-camera intrinsics + extrinsics, and per-arm
``hand_mat`` — all the extra inputs needed by the 3D-pos-encoding training
path.

Example usage:
  uv run examples/mesa/convert_mesa_data_to_lerobot.py \\
      --raw-dir /path/to/overfit --repo-id <org>/mesa-overfit
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

CAMERAS = ("egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand")
IMG_HW = (128, 128)
STATE_DIM = 14  # [joint_pos(6), grip(1)] per arm
ACTION_DIM = 14  # actions_joint_pos layout: [joint_pos(6), grip(1)] per arm

# MuJoCo depth linearization constants (stock Mesa sim setup).
# depth_m = ZNEAR * EXTENT / (1 - zbuffer * (1 - ZNEAR/ZFAR))
DEPTH_ZNEAR = 0.001
DEPTH_ZFAR = 50.0
DEPTH_EXTENT = 11.831


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    fps: int,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    include_3d: bool = False,
) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, *IMG_HW),
            "names": ["channels", "height", "width"],
        }
    if include_3d:
        for cam in CAMERAS:
            features[f"observation.depth.{cam}"] = {
                "dtype": "float32",
                "shape": (1, *IMG_HW),
                "names": ["channels", "height", "width"],
            }
            features[f"observation.intrinsic.{cam}"] = {
                "dtype": "float32",
                "shape": (3, 3),
                "names": None,
            }
            features[f"observation.extrinsic.{cam}"] = {
                "dtype": "float32",
                "shape": (4, 4),
                "names": None,
            }
        for arm in (0, 1):
            features[f"observation.hand_mat.robot{arm}"] = {
                "dtype": "float32",
                "shape": (4, 4),
                "names": None,
            }

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type="mesa_bimanual",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def linearize_depth(zbuffer: np.ndarray) -> np.ndarray:
    """MuJoCo z-buffer (in [~0.98, ~1.0]) -> metric depth in meters.

    Tabletop values land in ~[0.05, 3.0] m with the Mesa sim's znear/zfar/extent.
    """
    return DEPTH_ZNEAR * DEPTH_EXTENT / (1.0 - zbuffer * (1.0 - DEPTH_ZNEAR / DEPTH_ZFAR))


def discover_demos(raw_dir: Path, task: str | None) -> list[tuple[str, Path, str]]:
    """Return a list of ``(task_name, hdf5_path, demo_key)`` tuples.

    Accepts either a single HDF5 file or a directory laid out as
    ``<raw_dir>/<task>/demo/demo.hdf5``.
    """
    if raw_dir.is_file():
        task_name = task or raw_dir.parent.parent.name
        files = [(task_name, raw_dir)]
    else:
        files = [(p.parent.parent.name, p) for p in sorted(raw_dir.glob("*/demo/demo.hdf5"))]
        if not files:
            raise FileNotFoundError(f"No <task>/demo/demo.hdf5 found under {raw_dir}")

    demos: list[tuple[str, Path, str]] = []
    for task_name, hdf5_path in files:
        with h5py.File(hdf5_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        demos.extend((task_name, hdf5_path, k) for k in keys)
    return demos


def build_state(ep: h5py.Group) -> np.ndarray:
    """[joint_pos(6), grip(1)] per arm, mirroring the ``actions_joint_pos`` layout."""
    parts = []
    for arm in (0, 1):
        joint_pos = np.asarray(ep[f"obs/robot{arm}_joint_pos"], dtype=np.float32)
        grip = np.asarray(ep[f"obs/robot{arm}_gripper_jaw_width"], dtype=np.float32)[:, None]
        parts.extend([joint_pos, grip])
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_demo(hdf5_path: Path, demo_key: str, include_3d: bool = False) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        ep = f[f"data/{demo_key}"]
        out: dict[str, object] = {
            "state": build_state(ep),
            "action": np.asarray(ep["actions_joint_pos"], dtype=np.float32),
            "images": {cam: np.asarray(ep[f"obs/{cam}_image"]) for cam in CAMERAS},
        }
        if include_3d:
            out["depth"] = {
                cam: linearize_depth(np.asarray(ep[f"obs/{cam}_depth"], dtype=np.float32))
                for cam in CAMERAS
            }
            out["intrinsic"] = {
                cam: np.asarray(ep[f"obs/{cam}_intrinsic"], dtype=np.float32) for cam in CAMERAS
            }
            out["extrinsic"] = {
                cam: np.asarray(ep[f"obs/{cam}_extrinsic"], dtype=np.float32) for cam in CAMERAS
            }
            out["hand_mat"] = {
                arm: np.asarray(ep[f"obs/robot{arm}_hand_mat"], dtype=np.float32) for arm in (0, 1)
            }
        return out


def populate_dataset(
    dataset: LeRobotDataset,
    demos: list[tuple[str, Path, str]],
    include_3d: bool = False,
) -> LeRobotDataset:
    for task_name, hdf5_path, demo_key in tqdm.tqdm(demos):
        data = load_demo(hdf5_path, demo_key, include_3d=include_3d)
        for i in range(data["state"].shape[0]):
            frame = {
                "observation.state": data["state"][i],
                "action": data["action"][i],
                "task": task_name,
            }
            for cam in CAMERAS:
                frame[f"observation.images.{cam}"] = data["images"][cam][i]
            if include_3d:
                for cam in CAMERAS:
                    # HDF5 depth is (T, H, W, 1) HWC — transpose to (1, H, W) CHW to match feature shape.
                    frame[f"observation.depth.{cam}"] = np.transpose(data["depth"][cam][i], (2, 0, 1))
                    frame[f"observation.intrinsic.{cam}"] = data["intrinsic"][cam][i]
                    frame[f"observation.extrinsic.{cam}"] = data["extrinsic"][cam][i]
                for arm in (0, 1):
                    frame[f"observation.hand_mat.robot{arm}"] = data["hand_mat"][arm][i]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def port_mesa(
    raw_dir: Path,
    repo_id: str,
    *,
    task: str | None = None,
    fps: int = 20,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    include_3d: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    demos = discover_demos(raw_dir, task)
    dataset = create_empty_dataset(
        repo_id, fps=fps, mode=mode, dataset_config=dataset_config, include_3d=include_3d
    )
    populate_dataset(dataset, demos, include_3d=include_3d)
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_mesa)
