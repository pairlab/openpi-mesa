"""pi0/pi0.5 variant that swaps SigLIP's 2D learned pos-embed for 3D RoPE on image tokens
keyed by per-patch xyz derived from depth + camera calibration.

Per-patch xyz is lifted from depth using camera intrinsics + extrinsics, transformed into
the end-effector frame, resized to the SigLIP patch grid, and forwarded to the LLM as
``xyz_positions``. The Gemma attention layers rotate Q and K of those slots by 3D RoPE
(see :func:`openpi.models.gemma._apply_rope`); language and action tokens keep standard
1D Llama RoPE keyed by their integer sequence position.

The ``Pi0Adapt3RBimanual`` subclass extends this scheme to action tokens: each diffusion
chunk-step is split into two per-arm tokens (one per robot), each tagged with the arm's
end-effector position in the reference arm's eecf frame. This makes image↔action
cross-attention factorize through the same 3D relative-position bias that image↔image
attention uses. Within-arm temporal differentiation is provided by an additive learned
``temporal_pos_embed`` so that all 50 chunk-steps of one arm — which share the same xyz
RoPE key — remain distinguishable in attention.
"""

import dataclasses
import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import Literal, override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import Pi0, drop_one_camera_per_sample, make_attn_mask, posemb_sincos
from openpi.models.utils import point_cloud_utils as _pcu
from openpi.shared import array_typing as at
from openpi.training import key_utils as _key_utils

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0Adapt3RConfig(pi0_config.Pi0Config):
    """Config for the 3D-positional variant.

    Inherits every Pi0 knob (``pi05``, LoRA variants, action dims, token length, etc.).
    ``camera_keys`` and ``camera_drop_prob`` are inherited from :class:`Pi0Config`; the point
    cloud stack here scales with ``resolved_camera_keys`` the same way the RGB prefix does.

    ``xyz_rope_scale`` brings end-effector-frame meters (~[-0.5, 0.5]) into a frequency band
    the Gemma RoPE timescales can resolve. Without it, sub-meter xyz only excites the
    highest-frequency RoPE dims and most of the spectrum stays static across the workspace.
    """

    pc_frame: Literal["eecf"] = "eecf"
    pc_crop_radius: float = 0.0
    num_registers: int = 0
    xyz_rope_scale: float = 40.0

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

        camera_keys = self.resolved_camera_keys
        with at.disable_typechecking():
            observation_spec = _model.DepthObservation(
                images={k: image_spec for k in camera_keys},
                image_masks={k: image_mask_spec for k in camera_keys},
                depth_images={_key_utils.get_depth_key(_camera_name(k)): depth_spec for k in camera_keys},
                calibration={
                    **{_key_utils.get_intrinsics_key(_camera_name(k)): intrinsics_spec for k in camera_keys},
                    **{_key_utils.get_extrinsics_key(_camera_name(k)): extrinsics_spec for k in camera_keys},
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
    """Pi0 with per-patch 3D RoPE on image tokens, keyed by end-effector-frame xyz."""

    def __init__(self, config: Pi0Adapt3RConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        if config.pc_frame != "eecf":
            raise NotImplementedError(f"Only pc_frame='eecf' is supported, got {config.pc_frame!r}.")
        if config.pc_crop_radius != 0.0:
            raise NotImplementedError("pc_crop_radius > 0 is not supported in this trimmed port.")
        if config.num_registers != 0:
            raise NotImplementedError("num_registers > 0 is not supported in this trimmed port.")
        if config.xyz_rope_scale <= 0:
            raise ValueError("xyz_rope_scale must be > 0 for Pi0Adapt3R.")

        self.xyz_rope_scale = float(config.xyz_rope_scale)
        self.num_image_patches = len(config.resolved_camera_keys) * _PATCHES_PER_CAMERA

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

    def get_patch_xyz(
        self, observation: _model.PointCloudObservation
    ) -> at.Float[at.Array, "b t 3"]:
        camera_names = _key_utils.get_camera_names(observation)
        pcds = jnp.stack(
            [observation.point_clouds[_key_utils.get_point_cloud_key(n)] for n in camera_names], axis=1
        )
        B, ncam = pcds.shape[:2]
        pcds_resized = jax.image.resize(pcds, (B, ncam, _SPATIAL_GRID, _SPATIAL_GRID, 3), method="nearest")
        return einops.rearrange(pcds_resized, "b ncam h w d -> b (ncam h w) d")

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, dropout_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        observation = self.add_point_clouds_and_preprocess(observation, preprocess_rng, train=train)

        if train and self.camera_drop_prob > 0.0 and len(observation.image_masks) > 1:
            new_masks = drop_one_camera_per_sample(
                dropout_rng, observation.image_masks, self.camera_drop_prob
            )
            observation = observation.replace(image_masks=new_masks)

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

        positions, xyz_positions = self._build_xyz_positions(observation, positions, positions.shape[1])

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            xyz_positions=xyz_positions,
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

        positions, xyz_positions = self._build_xyz_positions(observation, positions, positions.shape[1])

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            xyz_positions=xyz_positions,
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

    def _build_xyz_positions(
        self,
        observation: _model.PointCloudObservation,
        positions: at.Int[at.Array, "b t"],
        total_seq_len: int,
    ) -> tuple[at.Int[at.Array, "b t"], at.Float[at.Array, "b t 3"]]:
        """Mark image-patch slots as 3D-RoPE (positions == -1) and build the per-slot xyz.

        ``xyz_positions`` is zero-padded outside the image block; the gemma rope kernel
        only consults it where ``positions == -1``, so the zero tail is inert for language,
        state, and action tokens.
        """
        patch_xyz = self.get_patch_xyz(observation) * self.xyz_rope_scale
        positions = positions.at[:, : self.num_image_patches].set(-1)
        pad = total_seq_len - patch_xyz.shape[1]
        if pad:
            patch_xyz = jnp.pad(patch_xyz, ((0, 0), (0, pad), (0, 0)))
        return positions, patch_xyz.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Phase 2: per-arm action tokens with 3D RoPE keyed by per-arm gripper xyz.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pi0Adapt3RBimanualConfig(Pi0Adapt3RConfig):
    """Config for Phase 2 — per-arm action token split with 3D RoPE.

    Two changes vs :class:`Pi0Adapt3RConfig`:

    1. The action chunk ``[B, action_horizon, action_dim]`` is split into two per-arm
       sub-chunks of width ``per_arm_action_dim`` each (default ``action_dim // 2``). Each
       per-arm sub-chunk is projected into its own action-expert tokens, doubling the
       suffix length to ``2 * action_horizon``.
    2. Action tokens are rotated by 3D RoPE keyed by per-arm gripper xyz in the reference
       arm's eecf frame: arm 0 lives at the origin, arm ``secondary_arm`` is placed at the
       relative offset between the two arms' ``hand_mat``. Within-arm temporal order is
       restored by an additive learned ``temporal_pos_embed[action_horizon, width]``
       added to each per-arm action token before the LLM (since all 50 tokens of one arm
       share the same RoPE key, attention cannot order them via RoPE alone).

    The data pipeline must populate ``observation.calibration["hand_mat_robot{secondary_arm}"]``
    in addition to the standard ``hand_mat`` / ``hand_mat_inv`` (set ``extra_arms`` on
    :class:`MESABimanualAdapt3RMultiCamDataConfig` accordingly).
    """

    secondary_arm: int = 1
    # When unset, defaults to ``action_dim // 2`` in :meth:`__post_init__`.
    per_arm_action_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.per_arm_action_dim is None:
            if self.action_dim % 2 != 0:
                raise ValueError(
                    f"action_dim must be even when per_arm_action_dim is unset, got {self.action_dim}."
                )
            object.__setattr__(self, "per_arm_action_dim", self.action_dim // 2)
        if self.per_arm_action_dim <= 0:
            raise ValueError(f"per_arm_action_dim must be > 0, got {self.per_arm_action_dim!r}.")
        if 2 * self.per_arm_action_dim > self.action_dim:
            raise ValueError(
                f"2 * per_arm_action_dim ({2 * self.per_arm_action_dim}) must not exceed "
                f"action_dim ({self.action_dim})."
            )
        if self.secondary_arm == 0:
            raise ValueError("secondary_arm must differ from the reference arm (robot0).")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_ADAPT3R_BIMANUAL

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Adapt3RBimanual":
        return Pi0Adapt3RBimanual(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        # Inject the secondary arm's hand_mat alongside the reference-arm hand_mat that the
        # parent already declares. Inputs spec types are immutable structs, so rebuild the
        # calibration dict with the extra entry and replace the observation in place. The
        # typecheck shim is needed because ``replace`` re-runs the dataclass typecheck and
        # the spec values are ``ShapeDtypeStruct``, not real arrays.
        calibration = dict(observation_spec.calibration)
        calibration[f"hand_mat_robot{self.secondary_arm}"] = jax.ShapeDtypeStruct(
            [batch_size, 4, 4], jnp.float32
        )
        with at.disable_typechecking():
            observation_spec = observation_spec.replace(calibration=calibration)
        return observation_spec, action_spec


class Pi0Adapt3RBimanual(Pi0Adapt3R):
    """Pi0Adapt3R with per-arm action tokens and 3D RoPE on the action expert.

    The token layout for the action-expert suffix becomes::

        [arm0_t0, arm0_t1, ..., arm0_t{H-1}, arm1_t0, arm1_t1, ..., arm1_t{H-1}]

    All ``2 * action_horizon`` tokens share one causal block (``ar_mask =
    [True] + [False] * (2 * action_horizon - 1)``), matching the base pi0 semantic that
    every action token can attend to every other action token.

    Per-arm action sub-chunks are pulled by slicing the noisy action chunk on its last
    axis: arm ``a`` takes ``noisy_actions[..., a*per_arm_action_dim : (a+1)*per_arm_action_dim]``.
    For the standard mesa-bimanual layout (``[jp0(6), grip0(1), jp1(6), grip1(1), pad(18)]``)
    with ``per_arm_action_dim=7`` this routes ``[jp0, grip0]`` to arm 0 and ``[jp1, grip1]``
    to arm 1. The default ``action_dim // 2`` is *not* appropriate for that layout (it would
    place both robots' real action dims into arm 0 and assign arm 1 to pure padding), so any
    bimanual TrainConfig using this model must set ``per_arm_action_dim`` explicitly.
    """

    def __init__(self, config: Pi0Adapt3RBimanualConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # The base Pi0 action_in_proj (action_dim -> width) and action_out_proj
        # (width -> action_dim) take the full bimanual chunk and produce one token per
        # chunk-step. Phase 2 replaces them with per-arm projections that halve the input
        # width and double the token count. The originals are deleted from the param tree
        # so they neither load from pi05_base nor occupy memory.
        del self.action_in_proj
        del self.action_out_proj

        self.per_arm_action_dim = int(config.per_arm_action_dim)
        self.secondary_arm = int(config.secondary_arm)
        self.action_in_proj_per_arm = nnx.Linear(
            self.per_arm_action_dim, action_expert_config.width, rngs=rngs
        )
        self.action_out_proj_per_arm = nnx.Linear(
            action_expert_config.width, self.per_arm_action_dim, rngs=rngs
        )
        # Small-std Gaussian rather than zeros: all H tokens of one arm share an xyz RoPE
        # key, so within-arm temporal ordering relies entirely on this additive embedding.
        # Zero-init makes the H tokens identical at step 0, so attention has no per-step
        # signal until gradient noise breaks the symmetry — std=1e-2 seeds a usable temporal
        # ordering immediately while staying well below the post-projection token magnitude.
        self.temporal_pos_embed = nnx.Param(
            jax.random.normal(
                rngs.params(), (self.action_horizon, action_expert_config.width), dtype=jnp.float32
            )
            * 1e-2
        )
        self.num_action_tokens = 2 * self.action_horizon

    # ------------------------------------------------------------------
    # Per-arm action-token construction
    # ------------------------------------------------------------------

    def _split_per_arm(
        self, actions: at.Float[at.Array, "b ah ad"]
    ) -> tuple[at.Float[at.Array, "b ah pad"], at.Float[at.Array, "b ah pad"]]:
        """Slice the action chunk into per-arm sub-chunks ``arm0 = [..., :pad]`` and
        ``arm1 = [..., pad:2*pad]``."""
        pad = self.per_arm_action_dim
        return actions[..., :pad], actions[..., pad : 2 * pad]

    def _embed_arm_actions(
        self, actions: at.Float[at.Array, "b ah pad"]
    ) -> at.Float[at.Array, "b ah width"]:
        """Project a per-arm sub-chunk and inject the shared learned temporal embedding."""
        tokens = self.action_in_proj_per_arm(actions)
        return tokens + self.temporal_pos_embed.value[None, :, :].astype(tokens.dtype)

    def _embed_action_suffix(
        self,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], at.Float[at.Array, "b emb"]]:
        """Build the per-arm action-token suffix.

        Mirrors :meth:`Pi0.embed_suffix` but produces ``2 * action_horizon`` tokens by
        concatenating ``[arm0_t0..arm0_t{H-1}, arm1_t0..arm1_t{H-1}]``. Returns the same
        4-tuple shape (tokens, input_mask, ar_mask, adarms_cond) so the call sites in
        :meth:`compute_loss` and :meth:`sample_actions` only need to know about the new
        token count.

        Pi0Adapt3RBimanual only supports the pi05 path (adaRMS time conditioning). The
        non-pi05 state-token branch from base ``Pi0.embed_suffix`` is intentionally not
        ported — we'd need a different RoPE strategy for the prepended state token and
        none of our configs use it.
        """
        if not self.pi05:
            raise NotImplementedError(
                "Pi0Adapt3RBimanual currently only supports pi05=True (adaRMS time conditioning)."
            )

        arm0_actions, arm1_actions = self._split_per_arm(noisy_actions)
        arm0_tokens = self._embed_arm_actions(arm0_actions)
        arm1_tokens = self._embed_arm_actions(arm1_actions)
        action_tokens = jnp.concatenate([arm0_tokens, arm1_tokens], axis=1)

        # Diffusion-time conditioning is identical to base pi0.5: one time embedding per
        # batch element, broadcast as adaRMS conditioning at every action-expert layer.
        time_emb = posemb_sincos(
            timestep, self.action_in_proj_per_arm.out_features, min_period=4e-3, max_period=4.0
        )
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)

        batch_size = action_tokens.shape[0]
        input_mask = jnp.ones((batch_size, self.num_action_tokens), dtype=jnp.bool_)
        # All action tokens (both arms) share one causal block — same semantic as base
        # pi0 (within-chunk full attention), extended to the doubled token count.
        ar_mask = jnp.array([True] + [False] * (self.num_action_tokens - 1))
        return action_tokens, input_mask, ar_mask, time_emb

    def _arm_xyz(
        self, observation: _model.PointCloudObservation
    ) -> at.Float[at.Array, "b 2 3"]:
        """Return the two arms' end-effector positions in the reference arm's eecf frame.

        Arm 0 (the reference arm) is at the eecf origin by construction. The secondary
        arm's pose is ``hand_mat_robot0_inv @ hand_mat_robot{secondary_arm}``; we extract
        the translation column and stack as ``[B, 2, 3]``.
        """
        hand_mat_inv = observation.calibration["hand_mat_inv"]
        secondary_key = f"hand_mat_robot{self.secondary_arm}"
        if secondary_key not in observation.calibration:
            raise ValueError(
                f"Pi0Adapt3RBimanual requires '{secondary_key}' in observation.calibration; "
                "set extra_arms on the data config."
            )
        hand_mat_secondary = observation.calibration[secondary_key]
        secondary_in_eecf = jnp.einsum("bij,bjk->bik", hand_mat_inv, hand_mat_secondary)
        arm0_xyz = jnp.zeros_like(secondary_in_eecf[..., :3, 3])
        arm1_xyz = secondary_in_eecf[..., :3, 3]
        return jnp.stack([arm0_xyz, arm1_xyz], axis=1)

    @override
    def _build_xyz_positions(
        self,
        observation: _model.PointCloudObservation,
        positions: at.Int[at.Array, "b t"],
        total_seq_len: int,
    ) -> tuple[at.Int[at.Array, "b t"], at.Float[at.Array, "b t 3"]]:
        """Tag both image patches *and* per-arm action tokens with ``positions == -1``
        and lay out their xyz keys.

        Layout (along the sequence axis):

        ``[image_patches..., language..., state..., arm0_actions..., arm1_actions...]``

        Image patches occupy ``[0, num_image_patches)`` and carry per-patch xyz (Phase 1
        behaviour). Action tokens occupy the trailing ``2 * action_horizon`` slots and
        carry per-arm xyz broadcast over their ``action_horizon`` chunk-steps. The middle
        block (language + state) keeps integer positions and zero xyz, so 1D RoPE applies
        there.
        """
        patch_xyz = self.get_patch_xyz(observation) * self.xyz_rope_scale
        arm_xyz = self._arm_xyz(observation) * self.xyz_rope_scale  # [B, 2, 3]

        # Mark sentinel positions for the two 3D-RoPE blocks.
        positions = positions.at[:, : self.num_image_patches].set(-1)
        positions = positions.at[:, -self.num_action_tokens :].set(-1)

        batch_size = patch_xyz.shape[0]
        mid = total_seq_len - self.num_image_patches - self.num_action_tokens
        if mid < 0:
            raise ValueError(
                f"total_seq_len ({total_seq_len}) is smaller than image+action tokens "
                f"({self.num_image_patches + self.num_action_tokens})."
            )

        # Per-arm xyz is constant across that arm's H chunk-steps — this matches the
        # observation-time gripper pose; per-step xyz from FK on the noisy action chunk is
        # left for a future revision.
        arm0_xyz_block = jnp.broadcast_to(
            arm_xyz[:, 0:1, :], (batch_size, self.action_horizon, 3)
        )
        arm1_xyz_block = jnp.broadcast_to(
            arm_xyz[:, 1:2, :], (batch_size, self.action_horizon, 3)
        )
        zero_block = jnp.zeros((batch_size, mid, 3), dtype=patch_xyz.dtype)
        xyz_positions = jnp.concatenate(
            [patch_xyz, zero_block, arm0_xyz_block, arm1_xyz_block], axis=1
        )
        return positions, xyz_positions.astype(jnp.float32)

    # ------------------------------------------------------------------
    # Loss and inference
    # ------------------------------------------------------------------

    def _action_chunk_from_per_arm(
        self,
        arm0_pred: at.Float[at.Array, "b ah pad"],
        arm1_pred: at.Float[at.Array, "b ah pad"],
        target_shape: tuple[int, ...],
    ) -> at.Float[at.Array, "b ah ad"]:
        """Re-pack per-arm predictions into the original ``[B, H, action_dim]`` chunk.

        The first ``per_arm_action_dim`` slots come from arm 0, the next
        ``per_arm_action_dim`` from arm 1. Any remaining trailing slots (when
        ``2 * per_arm_action_dim < action_dim``) are zero-filled — those slots are pure
        padding both at training time (their target velocity is structured noise and the
        loss-mean over the action axis still penalizes mismatch, but the MESAOutputs head
        slices the first 14 dims at inference, so this padding never reaches the env).
        """
        pad = self.per_arm_action_dim
        if 2 * pad == target_shape[-1]:
            return jnp.concatenate([arm0_pred, arm1_pred], axis=-1)
        tail_zeros = jnp.zeros(
            (*target_shape[:-1], target_shape[-1] - 2 * pad), dtype=arm0_pred.dtype
        )
        return jnp.concatenate([arm0_pred, arm1_pred, tail_zeros], axis=-1)

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, dropout_rng, noise_rng, time_rng = jax.random.split(rng, 4)
        observation = self.add_point_clouds_and_preprocess(observation, preprocess_rng, train=train)

        if train and self.camera_drop_prob > 0.0 and len(observation.image_masks) > 1:
            new_masks = drop_one_camera_per_sample(
                dropout_rng, observation.image_masks, self.camera_drop_prob
            )
            observation = observation.replace(image_masks=new_masks)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        positions, xyz_positions = self._build_xyz_positions(observation, positions, positions.shape[1])

        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            xyz_positions=xyz_positions,
        )

        action_out = suffix_out[:, -self.num_action_tokens :]
        arm0_out = action_out[:, : self.action_horizon, :]
        arm1_out = action_out[:, self.action_horizon :, :]
        v_arm0 = self.action_out_proj_per_arm(arm0_out)
        v_arm1 = self.action_out_proj_per_arm(arm1_out)
        v_t = self._action_chunk_from_per_arm(v_arm0, v_arm1, target_shape=u_t.shape)

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
        prefix_len = prefix_tokens.shape[1]
        suffix_len = self.num_action_tokens

        # Build positions / xyz over the full prefix+suffix length so we can slice each
        # block out as needed: prefix for the kv-cache fill, suffix for each diffusion
        # step.
        full_positions = jnp.cumsum(
            jnp.concatenate([prefix_mask, jnp.ones((batch_size, suffix_len), dtype=jnp.bool_)], axis=1),
            axis=1,
        ) - 1
        full_positions, full_xyz_positions = self._build_xyz_positions(
            observation, full_positions, prefix_len + suffix_len
        )

        prefix_positions = full_positions[:, :prefix_len]
        prefix_xyz_positions = full_xyz_positions[:, :prefix_len, :]
        suffix_positions_static = full_positions[:, prefix_len:]
        suffix_xyz_positions = full_xyz_positions[:, prefix_len:, :]

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            xyz_positions=prefix_xyz_positions,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self._embed_action_suffix(
                x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_repeat = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_repeat, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_len + suffix_tokens.shape[1],
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions_static,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                xyz_positions=suffix_xyz_positions,
            )
            assert prefix_out is None
            action_out = suffix_out[:, -self.num_action_tokens :]
            arm0_out = action_out[:, : self.action_horizon, :]
            arm1_out = action_out[:, self.action_horizon :, :]
            v_arm0 = self.action_out_proj_per_arm(arm0_out)
            v_arm1 = self.action_out_proj_per_arm(arm1_out)
            v_t = self._action_chunk_from_per_arm(v_arm0, v_arm1, target_shape=x_t.shape)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
