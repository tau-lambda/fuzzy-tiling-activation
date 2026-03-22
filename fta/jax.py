"""JAX/Flax implementation of the Fuzzy Tiling Activation (FTA).

Based on the Fuzzy Tiling Activation from Pan, Banman & White (2021),
extended with a right tile that linearly transforms inputs above the
core tiling range (in effect, outside the core tiling range, this
implementation behaves as a shifted ReLU).

Reference: https://arxiv.org/abs/1911.08068
"""

from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class FTA(nn.Module):
    """Fuzzy Tiling Activation with an extra linear tile.

    Tiles the interval ``[-bound, bound]`` and produces a fuzzy, sparse
    encoding for each input feature.  Inputs above the tiling range are
    passed through a shifted linear tile; inputs below the tiling range
    are zeroed out.

    The output dimensionality per feature is ``num_tiles + 1`` (the extra
    tile is the linear one).

    Attributes
    ----------
    bound : float
        Upper bound of the tiling range; the lower bound is ``-bound``.
    spillover_base : float
        Base value used to derive the spillover (called "sparisty parameter"
        and denoted as *eta* in the paper). Higher spillover means lower
        sparsity.
    spillover_mode : str
        How ``spillover_base`` is converted to the spillover:

        * ``"derive_from_bound"``: ``bound / 2 ** spillover_base``
        * ``"derive_from_tile_width"``: ``tile_width * 1.5 ** spillover_base``
        * ``"raw"`` or None: use ``spillover_base`` directly.
    tile_width : float or None
        Width of each tile.  Provide either ``tile_width`` or
        ``num_tiles`` (if both are given, ``tile_width`` is used).
    num_tiles : int or None
        Number of tiles in the tiling.  Provide either ``num_tiles`` or
        ``tile_width`` (if both are given, ``tile_width`` is used).
    dtype : jnp.dtype
        Data type for computation (default: float32).
    """

    bound: float
    spillover_base: float
    spillover_mode: str
    tile_width: Optional[float] = None
    num_tiles: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        """Apply the activation.

        Parameters
        ----------
        z : jax.Array
            Input array with features in the last dimension.

        Returns
        -------
        jax.Array
            Shape ``(batch_size, (num_tiles + 1) * features)``.
        """
        if self.bound <= 0:
            raise ValueError(f"bound must be > 0, got {self.bound}")
        if not (self.tile_width or self.num_tiles):
            raise ValueError("must provide one of tile_width or num_tiles")
        if self.num_tiles is not None and self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.tile_width is not None and self.tile_width <= 0:
            raise ValueError(f"tile_width must be > 0, got {self.tile_width}")

        z = z.astype(self.dtype)

        # Resolve the tile width
        tile_w_py = (
            float(self.tile_width)
            if self.tile_width is not None
            else (2.0 * float(self.bound)) / float(self.num_tiles)
        )
        tile_w = jnp.asarray(tile_w_py, dtype=z.dtype)

        # Build the tiling
        tiling = jnp.arange(
            -float(self.bound),
            float(self.bound),
            tile_w_py,
            dtype=z.dtype
        )
        num_core_tiles = tiling.shape[0]

        # Resolve the spillover
        if self.spillover_mode == "derive_from_bound":
            spillover_val = (
                float(self.bound) / (2.0 ** float(self.spillover_base))
            )
        elif self.spillover_mode == "derive_from_tile_width":
            spillover_val = tile_w_py * (1.5 ** float(self.spillover_base))
        elif self.spillover_mode == "raw" or self.spillover_mode is None:
            if self.spillover_base < 0:
                raise ValueError(
                    f"spillover_base must be >= 0 when "
                    f"spillover_mode is 'raw' or None, "
                    f"got {self.spillover_base}"
                )
            spillover_val = float(self.spillover_base)
        else:
            raise ValueError(
                f"spillover_mode must be 'derive_from_bound', "
                f"'derive_from_tile_width', 'raw', or None, "
                f"got {self.spillover_mode!r}"
            )
        spillover = jnp.asarray(spillover_val, dtype=z.dtype)

        # Make broadcast helpers
        z_exp = jnp.expand_dims(z, axis=-2)  # (B, 1, F)
        t_exp = jnp.reshape(tiling, (1, num_core_tiles, 1))  # (1, N, 1)

        # Compute the core fuzzy tiling activation
        left_gap  = jnp.maximum(t_exp - z_exp, 0.0)
        right_gap = jnp.maximum(z_exp - tile_w - t_exp, 0.0)
        intermediate = left_gap + right_gap  # (B, N, F)
        fuzzy_ind = jnp.where(
            intermediate > spillover,
            jnp.asarray(1.0, z.dtype),
            intermediate
        )
        core_output = jnp.asarray(1.0, z.dtype) - fuzzy_ind  # (B, N, F)

        # Zero-out core tiles for z above upper bound
        upper_bound = tiling[-1] + tile_w
        inside_mask = (z <= upper_bound).astype(z.dtype)  # (B, F)
        core_output = core_output * jnp.expand_dims(inside_mask, axis=-2)

        # Compute the right tile
        right_tail = jnp.maximum(
            z - upper_bound,
            0.0
        ).astype(z.dtype)  # (B, F)
        right_tile = jnp.expand_dims(
            right_tail,
            axis=-2
        )  # (B, 1, F)
        right_tile = jnp.broadcast_to(
            right_tile,
            core_output[..., :1, :].shape
        )  # (B, 1, F)

        # Concatenate the core and right tiles, and flatten
        final = jnp.concatenate(
            [core_output, right_tile],
            axis=-2
        )  # (B, N+1, F)
        out = final.reshape(
            z.shape[:-1] + (final.shape[-2] * final.shape[-1],)
        )  # (B, (N+1)*F)
        return out

    @property
    def num_output_tiles(self) -> int:
        """Total number of tiles (including the rightmost linear one)"""
        return int(self.num_tiles) + 1
