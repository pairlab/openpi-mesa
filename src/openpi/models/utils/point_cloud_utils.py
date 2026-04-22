"""Point-cloud utilities used by the 3D position-encoding path of :class:`Pi0Adapt3R`.

Only the functions required by ``enc_mode="resize"`` + ``pc_frame="eecf"`` are vendored
here: depth unprojection, rigid transformation. No open3d/matplotlib dependencies.
"""

import einops
import jax.numpy as jnp


def depth2fgpcd_batch(
    depth: jnp.ndarray,
    intrinsics: jnp.ndarray,
    *,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Unproject a batch of depth images into per-camera 3D points.

    Args:
        depth: ``(B, ncam, H, W)`` metric depth.
        intrinsics: ``(B, ncam, 3, 3)`` camera intrinsics.
        keepdims: if True return ``(B, ncam, H, W, 3)``, else ``(B, ncam*H*W, 3)``.
    """
    B, ncam, h, w = depth.shape

    fx = intrinsics[..., 0, 0].reshape(B, ncam, 1, 1)
    fy = intrinsics[..., 1, 1].reshape(B, ncam, 1, 1)
    cx = intrinsics[..., 0, 2].reshape(B, ncam, 1, 1)
    cy = intrinsics[..., 1, 2].reshape(B, ncam, 1, 1)

    pos_x, pos_y = jnp.meshgrid(
        jnp.arange(w, dtype=jnp.float32),
        jnp.arange(h, dtype=jnp.float32),
        indexing="xy",
    )
    pos_x = jnp.broadcast_to(pos_x, (B, ncam, h, w))
    pos_y = jnp.broadcast_to(pos_y, (B, ncam, h, w))

    x = (pos_x - cx) * depth / fx
    y = (pos_y - cy) * depth / fy
    z = depth

    pcd = jnp.stack([x, y, z], axis=-1)  # (B, ncam, H, W, 3)
    if keepdims:
        return pcd
    return einops.rearrange(pcd, "b ncam h w c -> b (ncam h w) c")


def batch_transform_point_cloud(pcd: jnp.ndarray, transform: jnp.ndarray) -> jnp.ndarray:
    """Apply a 4x4 rigid transform to a point cloud with arbitrary leading batch dims.

    Args:
        pcd: ``(..., N, 3)`` points.
        transform: ``(..., 4, 4)`` transformation matrices.
    """
    ones = jnp.ones((*pcd.shape[:-1], 1), dtype=pcd.dtype)
    pcd_homo = jnp.concatenate([pcd, ones], axis=-1)
    transform = transform.astype(pcd.dtype)
    trans_pcd_homo = jnp.einsum("...nd,...id->...ni", pcd_homo, transform)
    return trans_pcd_homo[..., :3]


def lift_point_cloud_batch(
    depths: jnp.ndarray,
    intrinsics: jnp.ndarray,
    extrinsics: jnp.ndarray,
    *,
    keepdims: bool = False,
) -> jnp.ndarray:
    """Unproject depth to per-camera points then transform into the world frame.

    Args:
        depths: ``(B, ncam, H, W)`` metric depth.
        intrinsics: ``(B, ncam, 3, 3)``.
        extrinsics: ``(B, ncam, 4, 4)`` cam-to-world transforms.
        keepdims: if True return ``(B, ncam, H, W, 3)``, else ``(B, ncam*H*W, 3)``.
    """
    _, ncam, H, _ = depths.shape
    pcd = depth2fgpcd_batch(depths, intrinsics, keepdims=False)
    pcd = einops.rearrange(pcd, "b (ncam hw) c -> b ncam hw c", ncam=ncam)
    pcd = batch_transform_point_cloud(pcd, extrinsics)
    if keepdims:
        return einops.rearrange(pcd, "b ncam (h w) c -> b ncam h w c", h=H)
    return einops.rearrange(pcd, "b ncam hw c -> b (ncam hw) c")
