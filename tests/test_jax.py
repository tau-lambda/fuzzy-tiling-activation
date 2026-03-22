"""Tests for the JAX/Flax FTA implementation."""

import pytest
import jax
import jax.numpy as jnp

from fta.jax import FTA


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_fta(**kwargs):
    """Create an FTA with sensible defaults; override via kwargs."""
    defaults = dict(
        bound=2.0,
        spillover_base=2.0,
        spillover_mode="derive_from_bound",
        num_tiles=4,
    )
    defaults.update(kwargs)
    return FTA(**defaults)


def _apply(fta, z):
    """Init and apply in one step (FTA has no learnable params)."""
    rng = jax.random.PRNGKey(0)
    params = fta.init(rng, z)
    return fta.apply(params, z)


# -----------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------

class TestInputValidation:
    def test_bound_zero(self):
        fta = _make_fta(bound=0)
        z = jnp.ones((1, 1))
        with pytest.raises(ValueError, match="bound must be > 0"):
            _apply(fta, z)

    def test_bound_negative(self):
        fta = _make_fta(bound=-1.0)
        z = jnp.ones((1, 1))
        with pytest.raises(ValueError, match="bound must be > 0"):
            _apply(fta, z)

    def test_neither_tile_width_nor_num_tiles(self):
        fta = _make_fta(tile_width=None, num_tiles=None)
        z = jnp.ones((1, 1))
        with pytest.raises(
            ValueError, match="must provide one of"
        ):
            _apply(fta, z)

    def test_num_tiles_zero(self):
        # 0 is falsy, so hits the "must provide one of" check first
        fta = _make_fta(num_tiles=0, tile_width=None)
        z = jnp.ones((1, 1))
        with pytest.raises(ValueError, match="must provide one of"):
            _apply(fta, z)

    def test_num_tiles_negative(self):
        fta = _make_fta(num_tiles=-3)
        z = jnp.ones((1, 1))
        with pytest.raises(ValueError, match="num_tiles must be >= 1"):
            _apply(fta, z)

    def test_tile_width_zero(self):
        # 0 is falsy, so hits the "must provide one of" check first
        fta = _make_fta(tile_width=0, num_tiles=None)
        z = jnp.ones((1, 1))
        with pytest.raises(
            ValueError, match="must provide one of"
        ):
            _apply(fta, z)

    def test_tile_width_negative(self):
        fta = _make_fta(tile_width=-0.5, num_tiles=None)
        z = jnp.ones((1, 1))
        with pytest.raises(
            ValueError, match="tile_width must be > 0"
        ):
            _apply(fta, z)

    def test_invalid_spillover_mode(self):
        fta = _make_fta(spillover_mode="oops")
        z = jnp.ones((1, 1))
        with pytest.raises(ValueError, match="spillover_mode must be"):
            _apply(fta, z)

    def test_negative_spillover_base_raw(self):
        fta = _make_fta(spillover_mode="raw", spillover_base=-1.0)
        z = jnp.ones((1, 1))
        with pytest.raises(
            ValueError, match="spillover_base must be >= 0"
        ):
            _apply(fta, z)

    def test_negative_spillover_base_none(self):
        fta = _make_fta(spillover_mode=None, spillover_base=-0.5)
        z = jnp.ones((1, 1))
        with pytest.raises(
            ValueError, match="spillover_base must be >= 0"
        ):
            _apply(fta, z)

    def test_valid_raw_spillover(self):
        fta = _make_fta(spillover_mode="raw", spillover_base=0.5)
        z = jnp.ones((1, 1))
        _apply(fta, z)  # should not raise

    def test_valid_none_spillover(self):
        fta = _make_fta(spillover_mode=None, spillover_base=0.0)
        z = jnp.ones((1, 1))
        _apply(fta, z)  # should not raise


# -----------------------------------------------------------------------
# Output shape
# -----------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("batch,features,num_tiles", [
        (1, 1, 4), (1, 5, 4), (4, 1, 4), (4, 3, 4), (16, 8, 4),
        (3, 2, 1), (3, 2, 2), (3, 2, 8), (3, 2, 20),
    ])
    def test_shape(self, batch, features, num_tiles):
        fta = _make_fta(num_tiles=num_tiles)
        z = jnp.ones((batch, features))
        out = _apply(fta, z)
        assert out.shape == (batch, fta.num_output_tiles * features)

    def test_shape_with_tile_width(self):
        fta = _make_fta(tile_width=0.5, num_tiles=None, bound=1.0)
        z = jnp.ones((2, 3))
        out = _apply(fta, z)
        # bound=1, tile_width=0.5 => 4 core tiles + 1 right = 5
        assert out.shape == (2, 5 * 3)

    def test_shape_single_tile(self):
        fta = _make_fta(bound=0.5, num_tiles=1)
        z = jnp.ones((1, 1))
        out = _apply(fta, z)
        assert out.shape == (1, fta.num_output_tiles * 1)


# -----------------------------------------------------------------------
# Known input/output values
#
# Standard config: bound=2, num_tiles=4, spillover_mode=derive_from_bound,
# spillover_base=2 => tile_width=1.0, spillover=0.5
# -----------------------------------------------------------------------

class TestKnownOutputs:
    @pytest.fixture
    def fta(self):
        return _make_fta()

    def _check(self, fta, input_val, expected):
        z = jnp.array([[input_val]])
        out = _apply(fta, z)
        assert out.tolist()[0] == pytest.approx(expected, abs=1e-6)

    # Inputs well outside the tiling range
    def test_well_below(self, fta):
        self._check(fta, -5.0, [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_well_above(self, fta):
        self._check(fta, 5.0, [0.0, 0.0, 0.0, 0.0, 3.0])

    # Tile boundaries (internal)
    def test_at_neg_bound(self, fta):
        self._check(fta, -2.0, [1.0, 0.0, 0.0, 0.0, 0.0])

    def test_internal_boundary_neg1(self, fta):
        self._check(fta, -1.0, [1.0, 1.0, 0.0, 0.0, 0.0])

    def test_internal_boundary_0(self, fta):
        self._check(fta, 0.0, [0.0, 1.0, 1.0, 0.0, 0.0])

    def test_internal_boundary_pos1(self, fta):
        self._check(fta, 1.0, [0.0, 0.0, 1.0, 1.0, 0.0])

    def test_at_upper_bound(self, fta):
        self._check(fta, 2.0, [0.0, 0.0, 0.0, 1.0, 0.0])

    # Mid-tile (inside spillover zone of adjacent tile)
    def test_mid_tile_first(self, fta):
        self._check(fta, -1.5, [1.0, 0.5, 0.0, 0.0, 0.0])

    def test_mid_tile_last(self, fta):
        self._check(fta, 1.5, [0.0, 0.0, 0.5, 1.0, 0.0])

    # Quarter and three-quarter points within tiles
    def test_tile0_quarter(self, fta):
        self._check(fta, -1.75, [1.0, 0.0, 0.0, 0.0, 0.0])

    def test_tile0_three_quarter(self, fta):
        self._check(fta, -1.25, [1.0, 0.75, 0.0, 0.0, 0.0])

    def test_tile1_quarter(self, fta):
        self._check(fta, -0.75, [0.75, 1.0, 0.0, 0.0, 0.0])

    def test_tile1_three_quarter(self, fta):
        self._check(fta, -0.25, [0.0, 1.0, 0.75, 0.0, 0.0])

    def test_tile2_quarter(self, fta):
        self._check(fta, 0.25, [0.0, 0.75, 1.0, 0.0, 0.0])

    def test_tile2_three_quarter(self, fta):
        self._check(fta, 0.75, [0.0, 0.0, 1.0, 0.75, 0.0])

    def test_tile3_quarter(self, fta):
        self._check(fta, 1.25, [0.0, 0.0, 0.75, 1.0, 0.0])

    def test_tile3_three_quarter(self, fta):
        self._check(fta, 1.75, [0.0, 0.0, 0.0, 1.0, 0.0])

    # Spillover boundaries
    def test_spillover_boundary_left_of_tile0(self, fta):
        self._check(fta, -2.5, [0.5, 0.0, 0.0, 0.0, 0.0])

    def test_inside_spillover_left_of_tile0(self, fta):
        self._check(fta, -2.25, [0.75, 0.0, 0.0, 0.0, 0.0])

    def test_spillover_boundary_right_of_tile0(self, fta):
        self._check(fta, -0.5, [0.5, 1.0, 0.5, 0.0, 0.0])

    def test_inside_spillover_right_of_tile0(self, fta):
        self._check(fta, -0.75, [0.75, 1.0, 0.0, 0.0, 0.0])

    # Rightmost tile
    def test_rightmost_tile_small(self, fta):
        self._check(fta, 2.5, [0.0, 0.0, 0.0, 0.0, 0.5])

    def test_rightmost_tile_large(self, fta):
        self._check(fta, 10.0, [0.0, 0.0, 0.0, 0.0, 8.0])


class TestKnownOutputsSingleTile:
    """Single-tile config: bound=0.5, num_tiles=1, spillover=0.25 raw."""

    @pytest.fixture
    def fta(self):
        return _make_fta(
            bound=0.5, num_tiles=1,
            spillover_base=0.25, spillover_mode="raw"
        )

    def _check(self, fta, input_val, expected):
        z = jnp.array([[input_val]])
        out = _apply(fta, z)
        assert out.tolist()[0] == pytest.approx(expected, abs=1e-6)

    def test_below(self, fta):
        self._check(fta, -2.0, [0.0, 0.0])

    def test_at_lower_bound(self, fta):
        self._check(fta, -0.5, [1.0, 0.0])

    def test_mid(self, fta):
        self._check(fta, 0.0, [1.0, 0.0])

    def test_at_upper_bound(self, fta):
        self._check(fta, 0.5, [1.0, 0.0])

    def test_above(self, fta):
        self._check(fta, 2.0, [0.0, 1.5])


class TestKnownOutputsTileWidth:
    """Config with explicit tile_width: bound=1.0, tile_width=0.5,
    spillover_mode=derive_from_tile_width, spillover_base=1.0
    => spillover=0.75"""

    @pytest.fixture
    def fta(self):
        return _make_fta(
            bound=1.0, tile_width=0.5, num_tiles=None,
            spillover_base=1.0,
            spillover_mode="derive_from_tile_width"
        )

    def _check(self, fta, input_val, expected):
        z = jnp.array([[input_val]])
        out = _apply(fta, z)
        assert out.tolist()[0] == pytest.approx(expected, abs=1e-6)

    def test_at_neg_bound(self, fta):
        self._check(fta, -1.0, [1.0, 0.5, 0.0, 0.0, 0.0])

    def test_internal_boundary(self, fta):
        self._check(fta, -0.5, [1.0, 1.0, 0.5, 0.0, 0.0])

    def test_center(self, fta):
        self._check(fta, 0.0, [0.5, 1.0, 1.0, 0.5, 0.0])

    def test_at_upper_bound(self, fta):
        self._check(fta, 1.0, [0.0, 0.0, 0.5, 1.0, 0.0])

    def test_above(self, fta):
        self._check(fta, 2.0, [0.0, 0.0, 0.0, 0.0, 1.0])


# -----------------------------------------------------------------------
# Multiple features
# -----------------------------------------------------------------------

class TestMultipleFeatures:
    def test_two_features(self):
        fta = _make_fta()
        z = jnp.array([
            [-5.0, 0.0],
            [0.0, 5.0],
            [1.0, -1.0],
        ])
        out = _apply(fta, z)
        assert out.shape == (3, 10)
        expected = [
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 3],
            [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        ]
        for i in range(3):
            assert (
                out[i].tolist()
                == pytest.approx(expected[i], abs=1e-6)
            )


# -----------------------------------------------------------------------
# tile_width vs num_tiles equivalence
# -----------------------------------------------------------------------

class TestTileWidthNumTilesEquivalence:
    def test_equivalent_outputs(self):
        fta_nt = _make_fta(
            bound=2.0, num_tiles=4, tile_width=None,
            spillover_base=0.5, spillover_mode="raw"
        )
        fta_tw = _make_fta(
            bound=2.0, num_tiles=None, tile_width=1.0,
            spillover_base=0.5, spillover_mode="raw"
        )
        z = jnp.array([[0.3, -1.2], [1.7, -0.4], [0.0, 0.5]])
        out_nt = _apply(fta_nt, z)
        out_tw = _apply(fta_tw, z)
        assert out_nt.shape == out_tw.shape
        assert jnp.allclose(out_nt, out_tw, atol=1e-6)

    def test_tile_width_wins_when_both_given(self):
        fta = _make_fta(
            bound=2.0, num_tiles=8, tile_width=1.0,
            spillover_base=0.5, spillover_mode="raw"
        )
        z = jnp.ones((1, 1))
        out = _apply(fta, z)
        # tile_width=1.0 with bound=2 => 4 core tiles + 1 = 5
        assert out.shape == (1, 5)


# -----------------------------------------------------------------------
# Gradient correctness
# -----------------------------------------------------------------------

class TestGradients:
    def test_grad_finite_interior(self):
        """Gradients are finite for inputs in the interior of tiles."""
        fta = _make_fta()
        rng = jax.random.PRNGKey(0)
        z = jnp.array([[-1.75, 0.25], [-0.25, 3.0]])
        params = fta.init(rng, z)

        def loss_fn(z):
            return fta.apply(params, z).sum()

        grads = jax.grad(loss_fn)(z)
        assert grads.shape == z.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_grad_zero_below_range(self):
        """Gradient is zero for inputs well below the tiling range."""
        fta = _make_fta()
        rng = jax.random.PRNGKey(0)
        z = jnp.array([[-5.0, -4.0]])
        params = fta.init(rng, z)

        def loss_fn(z):
            return fta.apply(params, z).sum()

        grads = jax.grad(loss_fn)(z)
        assert jnp.allclose(grads, 0.0)

    def test_grad_one_in_rightmost_tile(self):
        """Gradient is 1 in the rightmost tile."""
        fta = _make_fta()
        rng = jax.random.PRNGKey(0)
        z = jnp.array([[5.0]])
        params = fta.init(rng, z)

        def loss_fn(z):
            return fta.apply(params, z).sum()

        grads = jax.grad(loss_fn)(z)
        assert grads[0, 0] == pytest.approx(1.0)

    def test_grad_matches_torch_interior(self):
        """JAX and PyTorch gradients match for interior points."""
        import torch
        from fta.torch import FTA as FTA_Torch

        # JAX
        fta_jax = _make_fta()
        rng = jax.random.PRNGKey(0)
        z_jax = jnp.array([[-1.75, 0.25], [-0.25, 3.0]])
        params = fta_jax.init(rng, z_jax)

        def loss_fn(z):
            return fta_jax.apply(params, z).sum()

        grads_jax = jax.grad(loss_fn)(z_jax)

        # PyTorch
        fta_torch = FTA_Torch(
            bound=2.0, spillover_base=2.0,
            spillover_mode="derive_from_bound",
            tile_width=None, num_tiles=4,
        )
        z_torch = torch.tensor(
            [[-1.75, 0.25], [-0.25, 3.0]], requires_grad=True,
        )
        out_torch = fta_torch(z_torch)
        out_torch.sum().backward()

        assert jnp.allclose(
            grads_jax,
            jnp.array(z_torch.grad.numpy()),
            atol=1e-5,
        )


# -----------------------------------------------------------------------
# Cross-implementation consistency
# -----------------------------------------------------------------------

class TestCrossImplementation:
    def test_outputs_match_standard_config(self):
        """JAX and PyTorch produce identical outputs for standard config."""
        import torch
        from fta.torch import FTA as FTA_Torch

        inputs = [
            [-5.0], [-2.5], [-2.0], [-1.5], [-1.0], [-0.75],
            [-0.5], [0.0], [0.5], [1.0], [1.5], [2.0], [5.0],
        ]

        fta_torch = FTA_Torch(
            bound=2.0, spillover_base=2.0,
            spillover_mode="derive_from_bound",
            tile_width=None, num_tiles=4,
        )
        fta_jax = _make_fta()

        z_torch = torch.tensor(inputs)
        z_jax = jnp.array(inputs)

        out_torch = fta_torch(z_torch).detach().numpy()
        out_jax = _apply(fta_jax, z_jax)

        assert jnp.allclose(
            jnp.array(out_torch), out_jax, atol=1e-6,
        )

    def test_outputs_match_multi_feature(self):
        """JAX and PyTorch produce identical outputs with multiple features."""
        import torch
        from fta.torch import FTA as FTA_Torch

        fta_torch = FTA_Torch(
            bound=2.0, spillover_base=2.0,
            spillover_mode="derive_from_bound",
            tile_width=None, num_tiles=4,
        )
        fta_jax = _make_fta()

        z_np = [[-5.0, 0.0], [0.0, 5.0], [1.0, -1.0]]
        z_torch = torch.tensor(z_np)
        z_jax = jnp.array(z_np)

        out_torch = fta_torch(z_torch).detach().numpy()
        out_jax = _apply(fta_jax, z_jax)

        assert jnp.allclose(
            jnp.array(out_torch), out_jax, atol=1e-6,
        )

    def test_outputs_match_tile_width_config(self):
        """JAX and PyTorch produce identical outputs with explicit tile_width."""
        import torch
        from fta.torch import FTA as FTA_Torch

        fta_torch = FTA_Torch(
            bound=1.0, spillover_base=1.0,
            spillover_mode="derive_from_tile_width",
            tile_width=0.5, num_tiles=None,
        )
        fta_jax = _make_fta(
            bound=1.0, spillover_base=1.0,
            spillover_mode="derive_from_tile_width",
            tile_width=0.5, num_tiles=None,
        )

        inputs = [[-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0]]
        z_torch = torch.tensor(inputs)
        z_jax = jnp.array(inputs)

        out_torch = fta_torch(z_torch).detach().numpy()
        out_jax = _apply(fta_jax, z_jax)

        assert jnp.allclose(
            jnp.array(out_torch), out_jax, atol=1e-6,
        )
