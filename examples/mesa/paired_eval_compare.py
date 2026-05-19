"""Paired-test analysis of 2D vs 3D offline eval dumps.

Loads the per-chunk MSE arrays produced by ``offline_eval_3d.py --dump-stats``
for matched (config, step) pairs and reports:

  * means + bootstrap 95% CI on overall MSE and first-step MSE
  * paired t and Wilcoxon signed-rank on the 3D - 2D differences
  * fraction of chunks where 3D beats 2D (sign rate)
  * trajectory of (mean MSE) vs step for both configs

Both runs at the same step share the same TorchDataLoader seed (config.seed=42)
and the same shuffle-flag / NUM_BATCHES / BATCH_SIZE, so the i-th chunk in the
2D dump corresponds to the same underlying LeRobot row as the i-th chunk in
the 3D dump. Pairing is therefore positional.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from scipy import stats


def _bootstrap_ci(arr: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        means[i] = arr[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _load(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _paired_report(label: str, a: np.ndarray, b: np.ndarray) -> None:
    """``a`` is 2D-baseline, ``b`` is 3D. Both shape (N,). Reports b - a stats."""
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = b - a
    n = len(diff)

    mean_a, mean_b = float(a.mean()), float(b.mean())
    rel = (mean_b - mean_a) / mean_a if mean_a != 0 else float("nan")
    a_lo, a_hi = _bootstrap_ci(a)
    b_lo, b_hi = _bootstrap_ci(b)
    d_lo, d_hi = _bootstrap_ci(diff)

    t_stat, t_p = stats.ttest_rel(b, a, alternative="less")  # 3D < 2D one-sided
    try:
        w_stat, w_p = stats.wilcoxon(b, a, alternative="less", zero_method="wilcox")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
    win_rate = float((b < a).mean())

    print(f"  [{label}]")
    print(f"    2D mean = {mean_a:.5f}   95% CI [{a_lo:.5f}, {a_hi:.5f}]   n={n}")
    print(f"    3D mean = {mean_b:.5f}   95% CI [{b_lo:.5f}, {b_hi:.5f}]   n={n}")
    print(f"    diff (3D - 2D) mean = {diff.mean():.5f}   95% CI [{d_lo:.5f}, {d_hi:.5f}]   "
          f"rel = {rel * 100:+.1f}%")
    print(f"    paired t (one-sided 3D<2D): t={t_stat:.3f}  p={t_p:.4g}")
    print(f"    Wilcoxon  (one-sided 3D<2D): W={w_stat:.3f}  p={w_p:.4g}")
    print(f"    chunks where 3D beats 2D: {win_rate * 100:.1f}%  ({int(win_rate * n)}/{n})")


def _step_label(p: pathlib.Path) -> int:
    # filename: stepNNNNN_b{B}_n{N}_s{S}_shuffle{0|1}.npz
    name = p.stem
    return int(name.split("_")[0].removeprefix("step"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--two-d-dir",
        default="experiments/offline_eval/mesa_bimanual_lora_2d_gen_ego_v1",
    )
    parser.add_argument(
        "--three-d-dir",
        default="experiments/offline_eval/mesa_bimanual_lora_3d_gen_ego_v1",
    )
    parser.add_argument("--steps", type=int, nargs="*", default=None,
                        help="Steps to compare. If omitted, uses intersection of available .npz files.")
    args = parser.parse_args()

    two_d = pathlib.Path(args.two_d_dir)
    three_d = pathlib.Path(args.three_d_dir)
    two_files = {_step_label(p): p for p in sorted(two_d.glob("step*.npz"))}
    three_files = {_step_label(p): p for p in sorted(three_d.glob("step*.npz"))}
    available = sorted(set(two_files) & set(three_files))
    if args.steps is not None:
        steps = [s for s in args.steps if s in available]
        missing = sorted(set(args.steps) - set(steps))
        if missing:
            print(f"warn: missing dumps for steps {missing}")
    else:
        steps = available

    if not steps:
        print(f"no matched .npz dumps under {two_d} and {three_d}")
        return 1

    print(f"comparing 2D dir = {two_d}")
    print(f"comparing 3D dir = {three_d}")
    print(f"matched steps: {steps}\n")

    print(f"{'step':>6} | {'2D MSE':>8} {'3D MSE':>8} {'rel%':>7} {'p_t':>10} {'p_W':>10} {'2D first':>9} {'3D first':>9} {'rel%':>7} {'p_t':>10}")
    print("-" * 110)
    for step in steps:
        a = _load(two_files[step])
        b = _load(three_files[step])
        a_mse, b_mse = a["per_chunk_mse"], b["per_chunk_mse"]
        a_first, b_first = a["per_chunk_first_mse"], b["per_chunk_first_mse"]
        rel_mse = (b_mse.mean() - a_mse.mean()) / a_mse.mean() * 100
        rel_first = (b_first.mean() - a_first.mean()) / a_first.mean() * 100
        _, p_t_mse = stats.ttest_rel(b_mse, a_mse, alternative="less")
        _, p_t_first = stats.ttest_rel(b_first, a_first, alternative="less")
        try:
            _, p_w_mse = stats.wilcoxon(b_mse, a_mse, alternative="less", zero_method="wilcox")
        except ValueError:
            p_w_mse = float("nan")
        print(f"{step:>6} | {a_mse.mean():>8.5f} {b_mse.mean():>8.5f} {rel_mse:>+7.1f} "
              f"{p_t_mse:>10.4g} {p_w_mse:>10.4g} "
              f"{a_first.mean():>9.5f} {b_first.mean():>9.5f} {rel_first:>+7.1f} {p_t_first:>10.4g}")

    print()
    for step in steps:
        a = _load(two_files[step])
        b = _load(three_files[step])
        print(f"=== step {step} ===")
        _paired_report("overall MSE", a["per_chunk_mse"], b["per_chunk_mse"])
        _paired_report("first-step MSE", a["per_chunk_first_mse"], b["per_chunk_first_mse"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
