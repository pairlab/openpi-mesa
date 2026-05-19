"""Pre-training sanity checks for ``pi05_mesa_bimanual_lora_3d``.

Runs two quick checks so we don't burn H100 hours on a broken config:

1. Dataloader shape/dtype/numerics — one batch, verify depth is float32 with
   finite values, intrinsics/extrinsics are finite, ``hand_mat @ hand_mat_inv``
   is close to identity, and post-resize image/depth shapes are correct.
2. Model forward smoke — instantiate ``Pi0Adapt3R`` (random init, no weight
   load), run a single ``compute_loss`` on the batch, confirm loss is finite.
   This exercises the new ``pos_encodings`` plumbing in gemma plus the
   point-cloud lift / harmonic enc / MLP head end-to-end.

Run on any GPU node — batch size is shrunk to 2 to keep memory modest.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


_EXPECTED_IMG_HW = (224, 224)


def _fmt(arr: np.ndarray) -> str:
    return f"shape={tuple(arr.shape)} dtype={arr.dtype}"


def check_batch(obs: _model.DepthObservation, actions: jax.Array) -> None:
    print(f"[batch] actions {_fmt(np.asarray(actions))} "
          f"min={float(jnp.min(actions)):.3f} max={float(jnp.max(actions)):.3f}")
    assert bool(jnp.all(jnp.isfinite(actions))), "actions has NaN/Inf"

    if not isinstance(obs, _model.DepthObservation):
        raise TypeError(
            f"Expected DepthObservation from the data pipeline, got {type(obs).__name__}. "
            f"MESABimanualAdapt3RInputs → DepthObservation.from_dict wiring is broken."
        )

    # Images
    assert obs.images, "observation.images is empty"
    for k, img in obs.images.items():
        arr = np.asarray(img)
        print(f"[batch] image[{k}] {_fmt(arr)} min={arr.min():.3f} max={arr.max():.3f}")
        assert arr.shape[-3:-1] == _EXPECTED_IMG_HW, (
            f"image[{k}] expected HW={_EXPECTED_IMG_HW}, got {arr.shape}"
        )
        assert arr.shape[-1] == 3, f"image[{k}] expected C=3, got {arr.shape}"
        assert np.all(np.isfinite(arr)), f"image[{k}] has NaN/Inf"

    # Depth
    assert obs.depth_images is not None, (
        "observation.depth_images is None — MESABimanualAdapt3RInputs wiring broken"
    )
    for k, d in obs.depth_images.items():
        arr = np.asarray(d)
        print(f"[batch] depth[{k}] {_fmt(arr)} min={arr.min():.3f} max={arr.max():.3f}")
        assert arr.dtype == np.float32, f"depth[{k}] dtype {arr.dtype} != float32"
        assert arr.shape[-3:-1] == _EXPECTED_IMG_HW, (
            f"depth[{k}] expected HW={_EXPECTED_IMG_HW}, got {arr.shape}"
        )
        assert arr.shape[-1] == 1, f"depth[{k}] expected C=1, got {arr.shape}"
        assert np.all(np.isfinite(arr)), f"depth[{k}] has NaN/Inf"

    # Calibration
    assert obs.calibration is not None, "observation.calibration is None"
    for k in ("hand_mat", "hand_mat_inv"):
        assert k in obs.calibration, f"calibration missing {k!r}"
    for k, v in obs.calibration.items():
        arr = np.asarray(v)
        print(f"[batch] calib[{k}] {_fmt(arr)}")
        assert np.all(np.isfinite(arr)), f"calibration[{k}] has NaN/Inf"

    # hand_mat @ hand_mat_inv ≈ I, elementwise.
    hm = np.asarray(obs.calibration["hand_mat"])
    hmi = np.asarray(obs.calibration["hand_mat_inv"])
    prod = np.einsum("...ij,...jk->...ik", hm, hmi)
    err = float(np.max(np.abs(prod - np.eye(4))))
    print(f"[batch] max |hand_mat @ hand_mat_inv - I| = {err:.2e}")
    assert err < 1e-3, f"hand_mat inverse check failed (max err {err:.2e})"

    print("[batch] ✓ shapes / dtypes / numerics look clean.")


def check_model_forward(config: _config.TrainConfig, obs, actions) -> None:
    print("[model] instantiating Pi0Adapt3R (random init, no weight load)…")
    model = config.model.create(jax.random.key(0))

    print("[model] running compute_loss (train=False, num_steps=1)…")
    loss = model.compute_loss(jax.random.key(1), obs, actions, train=False)
    loss_mean = float(jnp.mean(loss))
    print(f"[model] loss {_fmt(np.asarray(loss))} mean={loss_mean:.6f}")
    assert bool(jnp.all(jnp.isfinite(loss))), "loss has NaN/Inf"
    print("[model] ✓ forward pass finite.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_mesa_bimanual_lora_3d")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--skip-model", action="store_true",
                        help="Only run the dataloader check (faster; no GPU forward).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = _config.get_config(args.config_name)
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0)

    print(f"[sanity] config={config.name} batch_size={config.batch_size}")
    loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=False)
    batches = list(loader)
    assert len(batches) == 1, f"expected 1 batch, got {len(batches)}"
    obs, actions = batches[0]

    check_batch(obs, actions)

    if args.skip_model:
        print("[sanity] --skip-model set; skipping model forward.")
    else:
        check_model_forward(config, obs, actions)

    print("[sanity] all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
