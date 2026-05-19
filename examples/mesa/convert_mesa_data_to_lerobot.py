"""Convert Mesa bimanual HDF5 data to the LeRobot v3.0 dataset format.

Each HDF5 holds one or more demos under ``data/demo_N``. An episode is written
per demo. Actions are the 14-D ``actions_joint_pos`` (absolute joint-space
target: ``[joint_pos(6), grip(1)]`` per arm). State mirrors that layout,
derived from ``robot{0,1}_joint_pos`` and ``robot{0,1}_gripper_jaw_width``.

When ``--include-3d`` is set, also emits per-camera metric depth (linearized
from MuJoCo's z-buffer), per-camera intrinsics + extrinsics, and per-arm
``hand_mat`` — all the extra inputs needed by the 3D-pos-encoding training
path.

Supported ``--raw-dir`` layouts:
  - A single HDF5 file with one or more demos.
  - ``<raw_dir>/<task>/demo/demo.hdf5`` — aggregated per-task file (overfit set).
  - ``<raw_dir>/<task>/demo/tmp/*.hdf5`` — MimicGen output, one demo per file
    (each file holds a single ``data/demo_0``).

Example usage:
  uv run examples/mesa/convert_mesa_data_to_lerobot.py \\
      --raw-dir /path/to/overfit --repo-id <org>/mesa-overfit
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

DEFAULT_CAMERAS: tuple[str, ...] = ("egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand")
STATE_DIM = 14  # [joint_pos(6), grip(1)] per arm
ACTION_DIM = 14  # actions_joint_pos layout: [joint_pos(6), grip(1)] per arm

# MuJoCo depth linearization constants (stock Mesa sim setup).
# depth_m = ZNEAR * EXTENT / (1 - zbuffer * (1 - ZNEAR/ZFAR))
DEPTH_ZNEAR = 0.001
DEPTH_ZFAR = 50.0
DEPTH_EXTENT = 11.831

# Raw z-buffer is stored as either float32 in [0, 1] (overfit HDF5s) or uint16
# scaled to [0, 65535] (MimicGen HDF5s). Normalize before linearizing.
DEPTH_UINT16_MAX = np.float32(np.iinfo(np.uint16).max)


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
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    image_hw: tuple[int, int] = (128, 128),
) -> LeRobotDataset:
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, *image_hw),
            "names": ["channels", "height", "width"],
        }
    if include_3d:
        for cam in cameras:
            features[f"observation.depth.{cam}"] = {
                "dtype": "float32",
                "shape": (1, *image_hw),
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
    """MuJoCo z-buffer (normalized to [0, 1]) -> metric depth in meters.

    Tabletop values land in ~[0.05, 3.0] m with the Mesa sim's znear/zfar/extent.
    uint16-stored buffers must be divided by 65535 before calling; use
    :func:`normalize_depth_zbuffer` to handle that uniformly.
    """
    return DEPTH_ZNEAR * DEPTH_EXTENT / (1.0 - zbuffer * (1.0 - DEPTH_ZNEAR / DEPTH_ZFAR))


def normalize_depth_zbuffer(raw: np.ndarray) -> np.ndarray:
    """Cast a stored depth buffer to a float32 z-buffer in [0, 1].

    Overfit HDF5s store the z-buffer as float32 already; MimicGen's default
    HDF5 writer scales it to uint16. Handle both.
    """
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / DEPTH_UINT16_MAX
    return raw.astype(np.float32)


def discover_demos(raw_dirs: list[Path], task: str | None) -> list[tuple[str, Path, str]]:
    """Return a list of ``(task_name, hdf5_path, demo_key)`` tuples.

    Each entry in ``raw_dirs`` may be any of:
      - a single HDF5 file (task name taken from ``--task`` or the parent-parent dir),
      - ``<raw_dir>/<task>/demo/demo.hdf5`` (aggregated overfit layout), or
      - ``<raw_dir>/<task>/demo/tmp/*.hdf5`` (MimicGen per-demo layout, one demo per file).
      - a per-variant ``<variant>/demo/tmp/*.hdf5`` subtree when raw_dir itself names the variant.

    When ``task`` is provided it overrides the auto-detected task name for every demo.
    """
    files: list[tuple[str, Path]] = []
    for raw_dir in raw_dirs:
        if raw_dir.is_file():
            task_name = task or raw_dir.parent.parent.name
            files.append((task_name, raw_dir))
            continue
        local: list[tuple[str, Path]] = []
        # Pattern 1: raw_dir is a parent with multiple task subtrees
        local += [(p.parent.parent.name, p) for p in sorted(raw_dir.glob("*/demo/demo.hdf5"))]
        local += [(p.parent.parent.parent.name, p) for p in sorted(raw_dir.glob("*/demo/tmp/*.hdf5"))]
        # Pattern 2: raw_dir IS the variant dir
        if not local:
            local += [(raw_dir.name, raw_dir / "demo" / "demo.hdf5")] if (raw_dir / "demo" / "demo.hdf5").is_file() else []
            local += [(raw_dir.name, p) for p in sorted((raw_dir / "demo" / "tmp").glob("*.hdf5"))] if (raw_dir / "demo" / "tmp").is_dir() else []
        if not local:
            raise FileNotFoundError(
                f"No <task>/demo/demo.hdf5 or <task>/demo/tmp/*.hdf5 found under {raw_dir}"
            )
        files += local

    if task is not None:
        files = [(task, p) for _, p in files]

    demos: list[tuple[str, Path, str]] = []
    for task_name, hdf5_path in files:
        with h5py.File(hdf5_path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        demos.extend((task_name, hdf5_path, k) for k in keys)
    return demos


def detect_image_hw(demos: list[tuple[str, Path, str]], camera: str) -> tuple[int, int]:
    """Peek at the first demo's first camera image to determine (H, W)."""
    _, hdf5_path, demo_key = demos[0]
    with h5py.File(hdf5_path, "r") as f:
        shape = f[f"data/{demo_key}/obs/{camera}_image"].shape
    # shape is (T, H, W, 3)
    return int(shape[1]), int(shape[2])


def build_state(ep: h5py.Group) -> np.ndarray:
    """[joint_pos(6), grip(1)] per arm, mirroring the ``actions_joint_pos`` layout."""
    parts = []
    for arm in (0, 1):
        joint_pos = np.asarray(ep[f"obs/robot{arm}_joint_pos"], dtype=np.float32)
        grip = np.asarray(ep[f"obs/robot{arm}_gripper_jaw_width"], dtype=np.float32)[:, None]
        parts.extend([joint_pos, grip])
    return np.concatenate(parts, axis=1).astype(np.float32)


def load_demo(
    hdf5_path: Path,
    demo_key: str,
    *,
    include_3d: bool = False,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as f:
        ep = f[f"data/{demo_key}"]
        out: dict[str, object] = {
            "state": build_state(ep),
            "action": np.asarray(ep["actions_joint_pos"], dtype=np.float32),
            "images": {cam: np.asarray(ep[f"obs/{cam}_image"]) for cam in cameras},
        }
        if include_3d:
            out["depth"] = {
                cam: linearize_depth(normalize_depth_zbuffer(np.asarray(ep[f"obs/{cam}_depth"])))
                for cam in cameras
            }
            out["intrinsic"] = {
                cam: np.asarray(ep[f"obs/{cam}_intrinsic"], dtype=np.float32) for cam in cameras
            }
            out["extrinsic"] = {
                cam: np.asarray(ep[f"obs/{cam}_extrinsic"], dtype=np.float32) for cam in cameras
            }
            out["hand_mat"] = {
                arm: np.asarray(ep[f"obs/robot{arm}_hand_mat"], dtype=np.float32) for arm in (0, 1)
            }
        return out


def populate_dataset(
    dataset: LeRobotDataset,
    demos: list[tuple[str, Path, str]],
    *,
    include_3d: bool = False,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
) -> LeRobotDataset:
    for task_name, hdf5_path, demo_key in tqdm.tqdm(demos):
        data = load_demo(hdf5_path, demo_key, include_3d=include_3d, cameras=cameras)
        for i in range(data["state"].shape[0]):
            frame = {
                "observation.state": data["state"][i],
                "action": data["action"][i],
                "task": task_name,
            }
            for cam in cameras:
                frame[f"observation.images.{cam}"] = data["images"][cam][i]
            if include_3d:
                for cam in cameras:
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
    repo_id: str,
    *,
    raw_dir: Path | None = None,
    raw_dirs: list[Path] | None = None,
    task: str | None = None,
    fps: int = 20,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    include_3d: bool = False,
    cameras: list[str] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    cams: tuple[str, ...] = tuple(cameras) if cameras else DEFAULT_CAMERAS
    dirs: list[Path] = list(raw_dirs) if raw_dirs else []
    if raw_dir is not None:
        dirs.append(raw_dir)
    if not dirs:
        raise ValueError("Provide --raw-dir or --raw-dirs (at least one).")
    demos = discover_demos(dirs, task)
    image_hw = detect_image_hw(demos, cams[0])
    dataset = create_empty_dataset(
        repo_id,
        fps=fps,
        mode=mode,
        dataset_config=dataset_config,
        include_3d=include_3d,
        cameras=cams,
        image_hw=image_hw,
    )
    populate_dataset(dataset, demos, include_3d=include_3d, cameras=cams)
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_mesa)
