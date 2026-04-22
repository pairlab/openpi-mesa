"""pi0/pi0.5 variant that replaces SigLIP's 2D learned pos-embed with a 3D per-patch
spatial position encoding derived from per-camera depth + calibration.

This is a trimmed port of the upstream Adapt3R implementation. Only one encoding mode
(``enc_mode="resize"``), one injection path (``pos_insertion_scale``), and one point-cloud
frame (``pc_frame="eecf"``) are supported. The PointNet/sample/median encoders, KV-cache
modulation, SigLIP posemb scaling, crop-mask plumbing, and register tokens are all
deliberately dropped — revive them as needed when the feature set grows.
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import Literal, override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.adapt3r_nnx_modules import HarmonicEncoding
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.utils import point_cloud_utils as _pcu
from openpi.shared import array_typing as at
from openpi.training import key_utils as _key_utils

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0Adapt3RConfig(pi0_config.Pi0Config):
    """Config for the 3D-positional variant.

    Inherits every Pi0 knob (``pi05``, LoRA variants, action dims, token length, etc.).
    The 3D head parameters live outside the LoRA scope and will train freely under the
    standard Pi0 LoRA freeze filter.
    """

    enc_mode: Literal["resize"] = "resize"
    pos_insertion_scale: float = 0.1
    pc_frame: Literal["eecf"] = "eecf"
    pc_crop_radius: float = 0.0  # retained for parity; only 0.0 is supported here.
    num_registers: int = 0  # retained for parity; only 0 is supported here.

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_ADAPT3R

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Adapt3R":
        return Pi0Adapt3R(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        depth_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 1], jnp.float32)
        intrinsics_spec = jax.ShapeDtypeStruct([batch_size, 3, 3], jnp.float32)
        extrinsics_spec = jax.ShapeDtypeStruct([batch_size, 4, 4], jnp.float32)

        with at.disable_typechecking():
            observation_spec = _model.DepthObservation(
                images={k: image_spec for k in _model.IMAGE_KEYS},
                image_masks={k: image_mask_spec for k in _model.IMAGE_KEYS},
                depth_images={_key_utils.get_depth_key(_camera_name(k)): depth_spec for k in _model.IMAGE_KEYS},
                calibration={
                    **{_key_utils.get_intrinsics_key(_camera_name(k)): intrinsics_spec for k in _model.IMAGE_KEYS},
                    **{_key_utils.get_extrinsics_key(_camera_name(k)): extrinsics_spec for k in _model.IMAGE_KEYS},
                    "hand_mat": extrinsics_spec,
                    "hand_mat_inv": extrinsics_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec


def _camera_name(image_key: str) -> str:
    return _key_utils.get_camera_name(image_key)


# SigLIP uses a 14×14 patch on a 224×224 image, producing a 16×16 token grid per camera.
_SPATIAL_GRID = 16
_PATCHES_PER_CAMERA = _SPATIAL_GRID * _SPATIAL_GRID


class Pi0Adapt3R(Pi0):
    """Pi0 + per-patch 3D positional encoding injected additively in the LLM prefix."""

    def __init__(self, config: Pi0Adapt3RConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        if config.enc_mode != "resize":
            raise NotImplementedError(f"Only enc_mode='resize' is supported, got {config.enc_mode!r}.")
        if config.pc_frame != "eecf":
            raise NotImplementedError(f"Only pc_frame='eecf' is supported, got {config.pc_frame!r}.")
        if config.pc_crop_radius != 0.0:
            raise NotImplementedError("pc_crop_radius > 0 is not supported in this trimmed port.")
        if config.num_registers != 0:
            raise NotImplementedError("num_registers > 0 is not supported in this trimmed port.")
        if config.pos_insertion_scale <= 0:
            raise ValueError("pos_insertion_scale must be > 0 for Pi0Adapt3R.")

        paligemma_config = _gemma_width(config.paligemma_variant)
        pe_dim = paligemma_config  # width of the first expert embedding
        num_freqs = 6
        in_dim = 3 * 2 * num_freqs
        hidden_dim = pe_dim // 2

        self.harmonic_encoding = HarmonicEncoding(num_freqs)
        self.pos_embed_l1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.pos_embed_l2 = nnx.Linear(hidden_dim, pe_dim, rngs=rngs)
        self.pos_insertion_scale = nnx.Param(jnp.array(config.pos_insertion_scale))

        self.num_image_patches = len(_model.IMAGE_KEYS) * _PATCHES_PER_CAMERA

    def add_point_clouds_and_preprocess(
        self, observation: _model.DepthObservation, rng: at.KeyArrayLike | None, *, train: bool = False
    ) -> _model.PointCloudObservation:
        """Lift per-camera depth to a world-frame point cloud, transform into the
        end-effector frame (eecf) via ``hand_mat_inv``, then run :func:`preprocess_observation`
        so the geometric augmentations applied to RGB are also applied to the PC.
        """
        camera_names = _key_utils.get_camera_names(observation)
        depths = jnp.stack(
            [observation.depth_images[_key_utils.get_depth_key(n)] for n in camera_names], axis=1
        ).squeeze(-1)
        intrinsics = jnp.stack(
            [observation.calibration[_key_utils.get_intrinsics_key(n)] for n in camera_names], axis=1
        )
        extrinsics = jnp.stack(
            [observation.calibration[_key_utils.get_extrinsics_key(n)] for n in camera_names], axis=1
        )
        B, ncam, H, W = depths.shape

        # World-frame lift, then transform into the end-effector frame.
        pcds = _pcu.lift_point_cloud_batch(depths, intrinsics, extrinsics)
        pcds = _pcu.batch_transform_point_cloud(pcds, observation.calibration["hand_mat_inv"])
        pcds = einops.rearrange(pcds, "b (ncam h w) d -> b ncam h w d", ncam=ncam, h=H, w=W)

        point_cloud_dict = {
            _key_utils.get_point_cloud_key(name): pcds[:, i] for i, name in enumerate(camera_names)
        }

        obs_dict = observation.to_dict()
        obs_dict["point_clouds"] = point_cloud_dict
        pc_obs = _model.PointCloudObservation.from_dict(obs_dict)

        return _model.preprocess_observation(
            rng, pc_obs, train=train, image_keys=list(pc_obs.images.keys())
        )

    def get_spatial_position_encoding(
        self, observation: _model.PointCloudObservation
    ) -> at.Float[at.Array, "b t d"]:
        camera_names = _key_utils.get_camera_names(observation)
        pcds = jnp.stack(
            [observation.point_clouds[_key_utils.get_point_cloud_key(n)] for n in camera_names], axis=1
        )
        B, ncam = pcds.shape[:2]
        pcds_resized = jax.image.resize(pcds, (B, ncam, _SPATIAL_GRID, _SPATIAL_GRID, 3), method="nearest")
        patch_locs = einops.rearrange(pcds_resized, "b ncam h w d -> b (ncam h w) d")
        return self.pos_embed_l2(nnx.relu(self.pos_embed_l1(self.harmonic_encoding(patch_locs))))

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = self.add_point_clouds_and_preprocess(observation, preprocess_rng, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        positions, pcds_enc = self._build_pos_encoding(observation, prefix_tokens, positions)

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            pos_encodings=[pcds_enc, None],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = self.add_point_clouds_and_preprocess(observation, None, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        positions, pcds_enc = self._build_pos_encoding(observation, prefix_tokens, positions)

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            pos_encodings=[pcds_enc, None],
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_repeat = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_repeat, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            step_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=step_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _build_pos_encoding(
        self,
        observation: _model.PointCloudObservation,
        prefix_tokens: at.Float[at.Array, "b s d"],
        positions: at.Int[at.Array, "b t"],
    ) -> tuple[at.Int[at.Array, "b t"], at.Float[at.Array, "b s d"]]:
        """Mark image-patch slots for 3D positional injection and build the per-slot encoding.

        The returned ``pcds_enc`` is zero-padded on the right so it can be added directly to
        ``prefix_tokens``; the zero tail covers any language tokens that follow the image
        patches. The ``positions == -1`` mask in :func:`_apply_rope` ensures those zeros
        are ignored.
        """
        pcds_enc = self.get_spatial_position_encoding(observation)
        positions = positions.at[:, : self.num_image_patches].set(-1)
        pad = prefix_tokens.shape[1] - pcds_enc.shape[1]
        if pad:
            pcds_enc = jnp.pad(pcds_enc, ((0, 0), (0, pad), (0, 0)))
        pcds_enc = (self.pos_insertion_scale.value * pcds_enc).astype(prefix_tokens.dtype)
        return positions, pcds_enc


def _gemma_width(variant: str) -> int:
    """Helper to get the first-expert embedding dim without duplicating the Gemma config table."""
    from openpi.models import gemma as _gemma

    return _gemma.get_config(variant).width
