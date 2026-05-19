"""Offline prediction sanity check for a trained ``pi05_mesa_bimanual_lora_3d``
checkpoint.

Loads a step-N checkpoint, pulls a handful of batches through the same 3D data
pipeline used at training time, calls ``model.sample_actions``, and compares
predicted action chunks against ground truth. Overall MSE near zero (and sign
agreement near 1.0) on training batches means the model has fit the data.

Compares in the *normalized + delta-packed* 32-D action space the model outputs.
Only the first 14 dims are real; the rest are zero padding and are excluded from
the per-dim summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as _lerobot_dataset
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi import transforms as _transforms


_REAL_ACTION_DIM = 14  # state/action layout: [jp0(6), g0(1), jp1(6), g1(1)]


def _load_policy(config: _config.TrainConfig, ckpt_dir: pathlib.Path):
    params_path = ckpt_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"missing params under {ckpt_dir!r}")
    print(f"[eval] loading params from {params_path}")
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
    return model


def _per_dim_stats(err: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> str:
    """``err``, ``gt``, ``pred`` are (B, H, 14). Returns a compact per-dim
    summary with mae, p99, gt and pred std (normalized space). The mae/std
    ratio is the most useful single number when gt_std is non-trivial:
    <0.1 is good, >0.5 means the prediction is barely better than predicting
    the GT mean. When gt_std is near zero the dim is constant in this slice
    and mae/std is meaningless — compare gt_std to pred_std instead to see
    whether the model is correctly outputting near-constant values."""
    mae = np.mean(np.abs(err), axis=(0, 1))
    p99 = np.quantile(np.abs(err), 0.99, axis=(0, 1))
    gt_std = np.std(gt, axis=(0, 1))
    pred_std = np.std(pred, axis=(0, 1))
    ratio = mae / np.maximum(gt_std, 1e-6)
    parts = [
        f"d{i}: mae={mae[i]:.3f} p99={p99[i]:.3f} "
        f"gt_std={gt_std[i]:.3f} pred_std={pred_std[i]:.3f} mae/std={ratio[i]:.2f}"
        for i in range(err.shape[-1])
    ]
    return "\n    " + "\n    ".join(parts)


def _summarize(pred: np.ndarray, gt: np.ndarray, dump_path: pathlib.Path | None = None) -> None:
    """Both arrays are (B, H, 32) in normalized-delta-padded action space."""
    pred_r = pred[..., :_REAL_ACTION_DIM]
    gt_r = gt[..., :_REAL_ACTION_DIM]
    err = pred_r - gt_r

    mse_full = float(np.mean(err ** 2))
    mae_full = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    first_step_mse = float(np.mean(err[:, 0] ** 2))

    # Per-chunk metrics so a paired test is possible across runs that share the
    # same shuffle seed (each entry is one of B sampled action chunks).
    per_chunk_mse = np.mean(err ** 2, axis=(1, 2))           # (B,)
    per_chunk_first_mse = np.mean(err[:, 0] ** 2, axis=1)    # (B,)
    per_chunk_mae = np.mean(np.abs(err), axis=(1, 2))        # (B,)

    def _sign_match(d: int) -> float:
        return float(np.mean(np.sign(pred_r[..., d]) == np.sign(gt_r[..., d])))

    def _stats(name: str, arr: np.ndarray) -> str:
        return (f"{name}: mean={arr.mean():.4f} std={arr.std(ddof=1):.4f} "
                f"sem={arr.std(ddof=1) / np.sqrt(len(arr)):.4f} "
                f"median={np.median(arr):.4f} q25={np.quantile(arr, 0.25):.4f} "
                f"q75={np.quantile(arr, 0.75):.4f} n={len(arr)}")

    print(f"[eval] pred shape={pred_r.shape}  gt shape={gt_r.shape}")
    print(f"[eval] overall  MSE={mse_full:.4f}  MAE={mae_full:.4f}  max|err|={max_abs:.4f}")
    print(f"[eval] first-step MSE={first_step_mse:.4f}")
    print(f"[eval] gripper sign-match: left(d6)={_sign_match(6):.3f}  right(d13)={_sign_match(13):.3f}")
    print(f"[eval] per-chunk {_stats('mse', per_chunk_mse)}")
    print(f"[eval] per-chunk {_stats('first_mse', per_chunk_first_mse)}")
    print(f"[eval] per-dim (over all B*H):" + _per_dim_stats(err, gt_r, pred_r))

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dump_path,
            pred=pred_r.astype(np.float32),
            gt=gt_r.astype(np.float32),
            err=err.astype(np.float32),
            per_chunk_mse=per_chunk_mse.astype(np.float64),
            per_chunk_first_mse=per_chunk_first_mse.astype(np.float64),
            per_chunk_mae=per_chunk_mae.astype(np.float64),
        )
        print(f"[eval] dumped per-chunk arrays to {dump_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_mesa_bimanual_lora_3d")
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Step checkpoint directory, e.g. .../mesa_bimanual_lora_3d_v1/7000")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--num-sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the data loader so sampled batches are drawn across all "
             "episodes rather than the first num_batches sequentially.",
    )
    parser.add_argument(
        "--keep-cameras",
        default="",
        help=(
            "Comma-separated camera mask keys (e.g. 'egocentric_0_rgb') to keep "
            "enabled. All other cameras have their image_masks zeroed before "
            "sample_actions, simulating a single-camera rollout. Empty = use all."
        ),
    )
    parser.add_argument(
        "--dump-stats",
        default="",
        help=(
            "Path to a .npz file where per-chunk MSE / first-step MSE / pred / gt "
            "arrays are saved. Required for paired-test analysis across runs."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0)
    print(f"[eval] config={config.name} batch_size={config.batch_size} "
          f"num_batches={args.num_batches} num_sample_steps={args.num_sample_steps}")

    model = _load_policy(config, ckpt_dir)

    loader = _data_loader.create_data_loader(
        config, num_batches=args.num_batches, shuffle=args.shuffle, skip_norm_stats=False,
    )

    keep_cameras = tuple(c.strip() for c in args.keep_cameras.split(",") if c.strip())
    if keep_cameras:
        print(f"[eval] keep_cameras={keep_cameras} (other cameras will be masked)")

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    t0 = time.time()
    for i, (obs, actions) in enumerate(loader):
        t_batch = time.time()
        if keep_cameras:
            unknown = set(keep_cameras) - set(obs.image_masks.keys())
            if unknown:
                raise ValueError(
                    f"--keep-cameras references unknown mask keys {sorted(unknown)}; "
                    f"available: {sorted(obs.image_masks.keys())}"
                )
            new_masks = {
                name: mask if name in keep_cameras else jnp.zeros_like(mask)
                for name, mask in obs.image_masks.items()
            }
            obs = dataclasses.replace(obs, image_masks=new_masks)
        rng = jax.random.key(args.seed + i)
        pred = model.sample_actions(rng, obs, num_steps=args.num_sample_steps)
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(actions, dtype=np.float32)
        preds.append(pred)
        gts.append(gt)
        print(f"[eval] batch {i}: {time.time() - t_batch:.1f}s "
              f"pred[{i}] min={pred[..., :_REAL_ACTION_DIM].min():.3f} "
              f"max={pred[..., :_REAL_ACTION_DIM].max():.3f}")

    print(f"[eval] {args.num_batches} batches in {time.time() - t0:.1f}s")
    dump_path = pathlib.Path(args.dump_stats) if args.dump_stats else None
    _summarize(np.concatenate(preds, axis=0), np.concatenate(gts, axis=0), dump_path=dump_path)
    print("[eval] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
