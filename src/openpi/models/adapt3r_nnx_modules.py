"""Small NNX modules used by the 3D position-encoding path of :class:`Pi0Adapt3R`.

Only the pieces required by ``enc_mode="resize"`` are vendored here.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def harmonic_encoding(x: jax.Array, num_freqs: int) -> jax.Array:
    """Sine/cosine harmonic encoding of a point.

    Given ``x`` with trailing dim ``D``, returns shape ``(..., D * 2 * num_freqs)``.
    """
    freqs = 2.0 ** jnp.arange(num_freqs) * jnp.pi
    x_proj = x[..., None] * freqs
    emb = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    return emb.reshape(*x.shape[:-1], -1)


class HarmonicEncoding(nnx.Module):
    def __init__(self, num_freqs: int):
        self.num_freqs = num_freqs

    def __call__(self, x: jax.Array) -> jax.Array:
        return harmonic_encoding(x, self.num_freqs)
