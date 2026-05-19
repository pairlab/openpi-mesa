"""Smoke test that BDDL/init-state pairings load and ``env.reset_to`` succeeds.

Reproduces the exact path the bench takes
(``vla-benchmark/scripts/eval_server_parallel.py:233-257``) without spinning
up a policy server, so it can be run on a CPU login/interactive node before
burning a GPU slot. For each ``(task, split, instance_idx)`` combination it:

  1. Loads the parsed BDDL via ``EvalSet.get_parsed_problem``.
  2. Loads the matching init-state from the shared (split-agnostic) pool.
  3. Builds the env via ``mesa.make_env`` with the same camera args the
     ``sbatch_eval_camdrop_ego.sh`` wrapper passes.
  4. Calls ``env.reset_to(init_state)``.

A failure here means the BDDL and init-state pool drifted out of sync (e.g.
the 2026-04-23 train regen incident), and the GPU eval job will die on every
episode before the policy is queried. Run this from the vla-benchmark venv:

    /storage/project/r-agarg35-0/fchang40/venvs/vla-benchmark/bin/python \
        examples/mesa/smoke_test_eval_bddls.py \
        --eval-set mesa_bimanual \
        --tasks apple_tray_on lime_bowl_on \
        --splits eval overfit \
        --num-instances 5
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass


@dataclass
class Result:
    task: str
    split: str
    instance_idx: int
    ok: bool
    error: str | None


def _force_offscreen_gl_backend() -> None:
    """Pick a MuJoCo GL backend that works on headless CPU nodes.

    Robosuite/MuJoCo requires a GL context at env construction time (cameras
    are baked into the model). On PACE login nodes we have no real GPU but
    ``libEGL`` is present, so EGL is the most reliable default. The user can
    override by exporting ``MUJOCO_GL`` before launching.
    """

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _smoke_one(
    eval_set,
    task_idx: int,
    task_name: str,
    instance_idx: int,
    camera_names: tuple[str, ...],
    camera_height: int,
    camera_width: int,
    controller_type: str,
    control_delta: bool,
    robots: tuple[str, ...],
) -> Result:
    from mesa import make_env

    try:
        parsed_problem = eval_set.get_parsed_problem(task_idx, instance_idx)
        init_state = eval_set.get_init_state(task_idx, instance_idx)
        if init_state is None:
            return Result(
                task=task_name,
                split=eval_set.split,
                instance_idx=instance_idx,
                ok=False,
                error="no init_state file in pool (split-agnostic pool missing this idx)",
            )
        env = make_env(
            parsed_problem=parsed_problem,
            controller_type=controller_type,
            control_delta=control_delta,
            camera_heights=camera_height,
            camera_widths=camera_width,
            camera_names=list(camera_names),
            robots=list(robots),
        )
        env.reset_to(init_state)
        try:
            env.close()
        except Exception:
            pass
        return Result(task=task_name, split=eval_set.split, instance_idx=instance_idx, ok=True, error=None)
    except Exception as exc:  # noqa: BLE001 — smoke harness wants every failure
        tb = traceback.format_exc(limit=3)
        return Result(
            task=task_name,
            split=eval_set.split,
            instance_idx=instance_idx,
            ok=False,
            error=f"{type(exc).__name__}: {exc}\n{tb}",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-set", default="mesa_bimanual")
    p.add_argument("--tasks", nargs="+", default=["apple_tray_on", "lime_bowl_on"])
    p.add_argument("--splits", nargs="+", default=["eval", "overfit"])
    p.add_argument(
        "--num-instances",
        type=int,
        default=5,
        help="Smoke each split with the first N instance indices (0..N-1).",
    )
    p.add_argument(
        "--camera-names",
        nargs="+",
        default=["egocentric", "leftshoulder", "rightshoulder", "midshoulder"],
    )
    p.add_argument("--camera-height", type=int, default=128)
    p.add_argument("--camera-width", type=int, default=128)
    # Match launch_eval.py:179 (joint_pos) and :128-129 (ReverseMountedYam x2)
    # so this smoke exercises the same env build path as the GPU eval job.
    p.add_argument("--controller-type", default="joint_pos")
    p.add_argument("--control-delta", action="store_true")
    p.add_argument(
        "--robots",
        nargs="+",
        default=["ReverseMountedYam", "ReverseMountedYam"],
        help="Must match launch_eval.py defaults; default Panda has only 13 robot visual geoms which the bimanual BDDLs do not expect.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failing (task, split, idx).",
    )
    args = p.parse_args()

    _force_offscreen_gl_backend()

    from mesa.task_suites.eval_set import EvalSet

    cameras = tuple(args.camera_names)
    overall_ok = True
    results: list[Result] = []

    for split in args.splits:
        eval_set = EvalSet(eval_set_name=args.eval_set, split=split)
        # Restrict to the requested tasks (if they exist).
        for task_name in args.tasks:
            if task_name not in eval_set.tasks:
                results.append(
                    Result(
                        task=task_name,
                        split=split,
                        instance_idx=-1,
                        ok=False,
                        error=f"task not in eval_set {args.eval_set!r}",
                    )
                )
                overall_ok = False
                if args.fail_fast:
                    break
                continue
            task_idx = eval_set.tasks.index(task_name)
            for idx in range(args.num_instances):
                r = _smoke_one(
                    eval_set,
                    task_idx=task_idx,
                    task_name=task_name,
                    instance_idx=idx,
                    camera_names=cameras,
                    camera_height=args.camera_height,
                    camera_width=args.camera_width,
                    controller_type=args.controller_type,
                    control_delta=args.control_delta,
                    robots=tuple(args.robots),
                )
                results.append(r)
                status = "PASS" if r.ok else "FAIL"
                print(f"[{status}] split={split:<8s} task={task_name:<14s} idx={idx:03d}")
                if not r.ok:
                    overall_ok = False
                    print(r.error)
                    if args.fail_fast:
                        break
            if args.fail_fast and not overall_ok:
                break
        if args.fail_fast and not overall_ok:
            break

    n_pass = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_pass
    print()
    print(f"summary: {n_pass} pass, {n_fail} fail (across {len(results)} cases)")
    if n_fail:
        print("first failure detail:")
        for r in results:
            if not r.ok:
                print(f"  split={r.split} task={r.task} idx={r.instance_idx}")
                print(f"  error: {r.error}")
                break
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
