"""Render the egocentric camera under the OOD pose override to verify framing.

Builds the bench env from a single BDDL eval instance, then renders a strip
comparing:
  - in-distribution (BDDL) min/base/max draws (phi ∈ [-π/6, +π/6])
  - OOD override min/base/max draws (default phi = π/2 ± 0.2)

Use this before kicking off long eval rollouts to confirm the OOD cam still
frames the workspace.

Run inside the vla-benchmark venv (needs mujoco offscreen renderer + GPU):
    /storage/project/r-agarg35-0/fchang40/venvs/vla-benchmark/bin/python \
        examples/mesa/render_ood_ego_check.py \
        --task apple_tray_on \
        --instance 0 \
        --output ood_ego_check.png
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy

import cv2
import numpy as np

import mesa  # noqa: F401  (registers MimicLabs_* envs)
from mesa import make_env
from mesa.sim.envs.utils import sample_camera_pose
from mesa.task_suites.eval_set import EvalSet
from mesa.utils.eval_helpers import apply_camera_overrides, parse_camera_overrides


def _camera_id(sim, name: str) -> int:
    if hasattr(sim.model, "camera_name2id"):
        return sim.model.camera_name2id(name)
    import mujoco
    return mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 8 + 8, 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _corner_params(base: dict, sign: float) -> dict:
    """Min (-1), base (0), or max (+1) corner of (r, theta, phi)."""
    out = dict(base)
    for field in ("r", "theta", "phi"):
        radius = base.get(f"{field}_radius", 0.0) or 0.0
        out[field] = base[field] + sign * radius
    for k in ("r_radius", "theta_radius", "phi_radius",
             "tilt_radius", "pan_radius"):
        out[k] = None
    return out


def _render_three(env, base: dict, prefix: str, size: int) -> list[np.ndarray]:
    sim = env.sim
    cam_id = _camera_id(sim, "egocentric")
    workspace_offset = (
        tuple(env.workspace_offset)
        if hasattr(env, "workspace_offset") else (0.0, 0.0, 0.9)
    )
    tiles: list[np.ndarray] = []
    for sign, name in ((-1.0, "min"), (0.0, "base"), (1.0, "max")):
        params = _corner_params(base, sign)
        pos, quat = sample_camera_pose(table_offset=workspace_offset, **params)
        sim.model.cam_pos[cam_id] = pos
        sim.model.cam_quat[cam_id] = quat
        sim.forward()
        img = sim.render(camera_name="egocentric", width=size, height=size)
        img = np.flipud(img).copy()
        text = (f"{prefix} {name}  r={params['r']:.2f}  "
                f"th={params['theta']:.2f}  phi={params['phi']:+.2f}")
        tiles.append(_label(img, text))
    return tiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-set-name", default="mesa_bimanual")
    p.add_argument("--eval-split", default="eval")
    p.add_argument("--task", default="apple_tray_on")
    p.add_argument("--instance", type=int, default=0)
    p.add_argument("--size", type=int, default=320)
    p.add_argument(
        "--camera-overrides",
        default='{"egocentric":{"phi":1.5707963267948966,"phi_radius":0.2}}',
        help="JSON; identical to the launch_eval.py / sbatch_eval_ego_ood.sh flag.",
    )
    p.add_argument("--output", default="ood_ego_check.png")
    args = p.parse_args()

    eval_set = EvalSet(args.eval_set_name, split=args.eval_split)
    if args.task not in eval_set.tasks:
        raise SystemExit(
            f"Task {args.task!r} not in {args.eval_set_name}/{args.eval_split}; "
            f"available: {eval_set.tasks[:10]}..."
        )
    task_id = eval_set.tasks.index(args.task)

    parsed_problem_in = eval_set.get_parsed_problem(task_id, args.instance)
    init_state = eval_set.get_init_state(task_id, args.instance)
    overrides = parse_camera_overrides(args.camera_overrides)
    parsed_problem_out = apply_camera_overrides(parsed_problem_in, overrides)

    base_in = deepcopy(parsed_problem_in["camera"]["egocentric"])
    base_out = deepcopy(parsed_problem_out["camera"]["egocentric"])

    print(f"in-distribution egocentric: {base_in}")
    print(f"OOD-overridden egocentric:  {base_out}")

    # Build env once (cameras are baked into the model from parsed_problem,
    # but we'll override cam_pos/cam_quat per render). Use the OOD-modified
    # parsed_problem so the model includes the moved camera; for the
    # in-distribution renders we just override cam_pos back.
    env = make_env(
        parsed_problem=parsed_problem_out,
        controller_type="joint_pos",
        control_delta=False,
        camera_heights=args.size,
        camera_widths=args.size,
        camera_names=["egocentric"],
        robots=["ReverseMountedYam", "ReverseMountedYam"],
    )
    if init_state is not None:
        env.reset_to(init_state)
    else:
        env.stable_reset()

    in_tiles = _render_three(env, base_in, "in", args.size)
    out_tiles = _render_three(env, base_out, "OOD", args.size)
    grid = np.concatenate(
        [np.concatenate(in_tiles, axis=1), np.concatenate(out_tiles, axis=1)],
        axis=0,
    )
    cv2.imwrite(args.output, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Wrote {os.path.abspath(args.output)}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
