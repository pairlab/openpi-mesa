#!/usr/bin/env python3
"""Asyncio supervisor: starts the pi05 policy server and the vla-benchmark
eval client in their respective venvs and keeps them in sync.

Port is auto-allocated and passed to both children, so multiple instances can
run concurrently on the same node.

Short-circuits if `<video-out>/<eval-set>/<exp>/<variant>/statistics/final_summary.json`
already exists, so re-running is a no-op.
"""

import argparse
import asyncio
import os
import signal
import socket
import sys
from dataclasses import dataclass

RESET = "\033[0m"
COLORS = {
    ("eval", "OUT"): "\033[96m",
    ("eval", "ERR"): "\033[91;1m",
    ("serve", "OUT"): "\033[92m",
    ("serve", "ERR"): "\033[93m",
}

# State-key order that must match pi05_mesa_bimanual_lora training:
#   observation.state = [jp0(6), g0(1), jp1(6), g1(1)]  (14-D)
STATE_KEYS = [
    "robot0_joint_pos",
    "robot0_gripper_jaw_width",
    "robot1_joint_pos",
    "robot1_gripper_jaw_width",
]


@dataclass
class Child:
    name: str
    cmd: list[str]
    proc: asyncio.subprocess.Process | None = None


async def _pump(stream: asyncio.StreamReader, name: str, stream_type: str):
    color = COLORS.get((name, stream_type), "")
    while True:
        line = await stream.readline()
        if not line:
            break
        print(f"{color}[{name}][{stream_type}] {line.decode(errors='replace').rstrip()}{RESET}", flush=True)


async def start_child(child: Child, cwd: str | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    child.proc = await asyncio.create_subprocess_exec(
        *child.cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    assert child.proc.stdout and child.proc.stderr
    asyncio.create_task(_pump(child.proc.stdout, child.name, "OUT"))
    asyncio.create_task(_pump(child.proc.stderr, child.name, "ERR"))


async def terminate_child(child: Child, timeout_s: float = 5.0) -> None:
    if not child.proc or child.proc.returncode is not None:
        return
    child.proc.terminate()
    try:
        await asyncio.wait_for(child.proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        child.proc.kill()
        await child.proc.wait()


async def wait_port_listening(
    host: str, port: int, timeout_s: float, proc: asyncio.subprocess.Process,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while True:
        if proc.returncode is not None:
            raise RuntimeError(f"Policy server exited early with code {proc.returncode}")
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"Timed out waiting for {host}:{port}")
            await asyncio.sleep(0.25)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True,
                        help="pi05 checkpoint step dir (e.g. .../9000).")
    parser.add_argument("--config", default="pi05_mesa_bimanual_lora",
                        help="openpi TrainConfig name.")

    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--variant-name", required=True)
    parser.add_argument("--eval-set-name", default="mesa_bimanual")
    parser.add_argument("--eval-split", default="overfit")
    parser.add_argument("--task-filter", nargs="+", default=["apple_tray_on"])

    parser.add_argument("--num-rollouts-per-task", type=int, default=4)
    parser.add_argument("--num-env-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)

    parser.add_argument("--camera-names", nargs="+",
                        default=["egocentric", "robot0_eye_in_hand", "robot1_eye_in_hand"])
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--robots", nargs="+",
                        default=["ReverseMountedYam", "ReverseMountedYam"])
    parser.add_argument("--visualization-camera-name", default="egocentric")

    parser.add_argument("--video-out-path",
                        default=os.path.join(os.getcwd(), "experiments", "vla_benchmark"))

    args = parser.parse_args()

    summary_path = os.path.join(
        args.video_out_path, args.eval_set_name, args.exp_name, args.variant_name,
        "statistics", "final_summary.json",
    )
    if os.path.exists(summary_path):
        print(f"[launch] final_summary already exists at {summary_path}; nothing to do.")
        return 0

    port = find_free_port()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(os.path.dirname(script_dir))

    serve_cmd = [
        sys.executable, os.path.join(script_dir, "eval_policy_server.py"),
        "--config", args.config,
        "--checkpoint-dir", args.checkpoint_dir,
        "--port", str(port),
    ]

    eval_cmd = [
        os.path.join(script_dir, "run-vla-benchmark.sh"),
        f"--eval-set-name={args.eval_set_name}",
        f"--eval-split={args.eval_split}",
        f"--exp-name={args.exp_name}",
        f"--variant-name={args.variant_name}",
        f"--num-rollouts-per-task={args.num_rollouts_per_task}",
        f"--num-env-workers={args.num_env_workers}",
        f"--seed={args.seed}",
        f"--port={port}",
        f"--replan-steps={args.replan_steps}",
        f"--controller-type=joint_pos",
        f"--camera-height={args.camera_height}",
        f"--camera-width={args.camera_width}",
        f"--visualization-camera-name={args.visualization_camera_name}",
        f"--video-out-path={args.video_out_path}",
        "--camera-names", *args.camera_names,
        "--robots", *args.robots,
        "--state-keys", *STATE_KEYS,
        "--task-filter", *args.task_filter,
    ]
    if args.max_steps is not None:
        eval_cmd.append(f"--max-steps={args.max_steps}")

    # 3D configs (Pi0Adapt3R) need depth + calibration on the wire. Depth is
    # enabled via --camera-depths and must be transported in meters to match
    # training (see examples/mesa/convert_mesa_data_to_lerobot.py:linearize_depth).
    if args.config.endswith("_3d"):
        eval_cmd.append("--camera-depths")
        eval_cmd.append("--depth-transport=meters")

    print(f"[launch] serve_cmd = {' '.join(serve_cmd)}", flush=True)
    print(f"[launch] eval_cmd  = {' '.join(eval_cmd)}", flush=True)

    serve = Child("serve", serve_cmd)
    evaluator = Child("eval", eval_cmd)
    children = [serve, evaluator]

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    try:
        await start_child(serve, cwd=repo_dir)
        await wait_port_listening("127.0.0.1", port, timeout_s=300.0, proc=serve.proc)
        print(f"[launch] policy server listening on port {port}", flush=True)

        await start_child(evaluator, cwd=repo_dir)

        waiters = [asyncio.create_task(c.proc.wait()) for c in children]
        stop_task = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(waiters + [stop_task], return_when=asyncio.FIRST_COMPLETED)

        if stop_task in done:
            return 130
        rc_map = {c.name: c.proc.returncode for c in children}
        if any(rc and rc != 0 for rc in rc_map.values()):
            print(f"[launch] child failure: {rc_map}", flush=True)
            return 1
        return 0
    finally:
        for c in children:
            await terminate_child(c)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
