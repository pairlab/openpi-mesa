"""Unit tests for `Pi0Adapt3RBimanual`'s per-arm action split/recombine.

These tests pin the contract that the per-arm split routes one robot's real action dims
into one token and the other robot's into the other token. The split/recombine methods
only depend on `self.per_arm_action_dim`, so we exercise them via a `SimpleNamespace`
stub rather than instantiating the full Gemma-backed model.
"""

import types

import jax.numpy as jnp

from openpi.models.pi0_adapt3r import Pi0Adapt3RBimanual


def _marker_chunk() -> jnp.ndarray:
    """Return a `[1, 1, 32]` action chunk where the first 14 dims are unique markers
    `[1..14]` and the trailing 18 padding dims are zero — matching the mesa-bimanual
    layout `[jp0(6), grip0(1), jp1(6), grip1(1), pad(18)]`.
    """
    return jnp.zeros((1, 1, 32), dtype=jnp.float32).at[0, 0, :14].set(jnp.arange(1, 15.0))


def test_split_per_arm_routes_each_robot_to_its_own_token_when_pad_is_7():
    fake_self = types.SimpleNamespace(per_arm_action_dim=7)
    actions = _marker_chunk()

    arm0, arm1 = Pi0Adapt3RBimanual._split_per_arm(fake_self, actions)

    assert arm0.shape == (1, 1, 7)
    assert arm1.shape == (1, 1, 7)
    assert jnp.array_equal(arm0[0, 0], jnp.arange(1, 8.0)), "arm 0 must receive [jp0, grip0]"
    assert jnp.array_equal(arm1[0, 0], jnp.arange(8, 15.0)), "arm 1 must receive [jp1, grip1]"


def test_action_chunk_from_per_arm_round_trips_first_14_dims_when_pad_is_7():
    fake_self = types.SimpleNamespace(per_arm_action_dim=7)
    actions = _marker_chunk()
    arm0, arm1 = Pi0Adapt3RBimanual._split_per_arm(fake_self, actions)

    recombined = Pi0Adapt3RBimanual._action_chunk_from_per_arm(
        fake_self, arm0, arm1, target_shape=(1, 1, 32)
    )

    assert recombined.shape == (1, 1, 32)
    assert jnp.array_equal(recombined[0, 0, :14], actions[0, 0, :14])
    assert jnp.array_equal(recombined[0, 0, 14:], jnp.zeros(18))


def test_split_per_arm_default_pad_16_is_degenerate_for_mesa_layout():
    """Regression-pin: the historical default `per_arm_action_dim = action_dim // 2 = 16`
    silently routes both robots' real action dims into arm 0 and assigns arm 1 to predict
    pure padding. Documenting this so a future contributor doesn't reintroduce the default
    for a bimanual config without thinking through the action layout.
    """
    fake_self = types.SimpleNamespace(per_arm_action_dim=16)
    actions = _marker_chunk()

    arm0, arm1 = Pi0Adapt3RBimanual._split_per_arm(fake_self, actions)

    assert jnp.array_equal(arm0[0, 0, :14], jnp.arange(1, 15.0)), (
        "with pad=16, arm 0 absorbs both robots' action dims"
    )
    assert jnp.array_equal(arm0[0, 0, 14:], jnp.zeros(2))
    assert jnp.array_equal(arm1[0, 0], jnp.zeros(16)), (
        "with pad=16, arm 1 sees only padding zeros"
    )
