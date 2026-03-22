"""Tests for the PyTorch FTA implementation."""

import pytest
import torch
from torch.autograd import gradcheck

from fta.torch import FTA


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_fta(**kwargs):
    """Create an FTA with sensible defaults; override via kwargs."""
    defaults = dict(
        bound=2.0,
        spillover_base=2.0,
        spillover_mode="derive_from_bound",
        tile_width=None,
        num_tiles=4,
    )
    defaults.update(kwargs)
    return FTA(**defaults)


# -----------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------

class TestInputValidation:
    def test_bound_zero(self):
        with pytest.raises(ValueError, match="bound must be > 0"):
            _make_fta(bound=0)

    def test_bound_negative(self):
        with pytest.raises(ValueError, match="bound must be > 0"):
            _make_fta(bound=-1.0)

    def test_neither_tile_width_nor_num_tiles(self):
        with pytest.raises(
            ValueError, match="must provide one of"
        ):
            _make_fta(tile_width=None, num_tiles=None)

    def test_num_tiles_zero(self):
        # 0 is falsy, so hits the "must provide one of" check first
        with pytest.raises(ValueError, match="must provide one of"):
            _make_fta(num_tiles=0, tile_width=None)

    def test_num_tiles_negative(self):
        with pytest.raises(ValueError, match="num_tiles must be >= 1"):
            _make_fta(num_tiles=-3)

    def test_tile_width_zero(self):
        # 0 is falsy, so hits the "must provide one of" check first
        with pytest.raises(
            ValueError, match="must provide one of"
        ):
            _make_fta(tile_width=0, num_tiles=None)

    def test_tile_width_negative(self):
        with pytest.raises(
            ValueError, match="tile_width must be > 0"
        ):
            _make_fta(tile_width=-0.5, num_tiles=None)

    def test_invalid_spillover_mode(self):
        with pytest.raises(ValueError, match="spillover_mode must be"):
            _make_fta(spillover_mode="oops")

    def test_negative_spillover_base_raw(self):
        with pytest.raises(
            ValueError, match="spillover_base must be >= 0"
        ):
            _make_fta(spillover_mode="raw", spillover_base=-1.0)

    def test_negative_spillover_base_none(self):
        with pytest.raises(
            ValueError, match="spillover_base must be >= 0"
        ):
            _make_fta(spillover_mode=None, spillover_base=-0.5)

    def test_valid_raw_spillover(self):
        fta = _make_fta(spillover_mode="raw", spillover_base=0.5)
        assert fta.spillover == 0.5

    def test_valid_none_spillover(self):
        fta = _make_fta(spillover_mode=None, spillover_base=0.0)
        assert fta.spillover == 0.0


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
        z = torch.randn(batch, features)
        out = fta(z)
        assert out.shape == (batch, fta.num_tiles * features)

    def test_shape_with_tile_width(self):
        fta = _make_fta(tile_width=0.5, num_tiles=None, bound=1.0)
        z = torch.randn(2, 3)
        out = fta(z)
        assert out.shape == (2, fta.num_tiles * 3)

    def test_shape_single_tile(self):
        fta = _make_fta(bound=0.5, num_tiles=1)
        z = torch.randn(1, 1)
        out = fta(z)
        assert out.shape == (1, fta.num_tiles * 1)


# -----------------------------------------------------------------------
# Spillover modes
# -----------------------------------------------------------------------

class TestSpilloverModes:
    def test_derive_from_bound(self):
        fta = _make_fta(
            bound=4.0, spillover_base=2.0,
            spillover_mode="derive_from_bound"
        )
        assert fta.spillover == pytest.approx(4.0 / 2**2.0)

    def test_derive_from_tile_width(self):
        fta = _make_fta(
            bound=2.0, spillover_base=1.0,
            spillover_mode="derive_from_tile_width",
            tile_width=0.5, num_tiles=None
        )
        assert fta.spillover == pytest.approx(0.5 * 1.5**1.0)

    def test_raw(self):
        fta = _make_fta(spillover_mode="raw", spillover_base=0.3)
        assert fta.spillover == pytest.approx(0.3)

    def test_none_equivalent_to_raw(self):
        fta = _make_fta(spillover_mode=None, spillover_base=0.3)
        assert fta.spillover == pytest.approx(0.3)


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
        z = torch.tensor([[input_val]])
        out = fta(z)
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
        z = torch.tensor([[input_val]])
        out = fta(z)
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
        z = torch.tensor([[input_val]])
        out = fta(z)
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
        z = torch.tensor([
            [-5.0, 0.0],
            [0.0, 5.0],
            [1.0, -1.0],
        ])
        out = fta(z)
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
        z = torch.randn(5, 3)
        out_nt = fta_nt(z)
        out_tw = fta_tw(z)
        assert out_nt.shape == out_tw.shape
        assert torch.allclose(out_nt, out_tw, atol=1e-6)

    def test_tile_width_wins_when_both_given(self):
        fta = _make_fta(
            bound=2.0, num_tiles=8, tile_width=1.0,
            spillover_base=0.5, spillover_mode="raw"
        )
        assert fta.tile_width == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Gradient correctness
# -----------------------------------------------------------------------

class TestGradients:
    def test_gradcheck_interior_points(self):
        """Analytical and numerical gradients agree for interior points.

        With bound=2, tile_width=1, spillover=0.5, discontinuities
        are at: -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2.
        """
        fta = _make_fta()
        fta._tiling = fta._tiling.double()
        z = torch.tensor(
            [[-1.75, 0.25], [-0.25, 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert gradcheck(fta, (z,), eps=1e-6, atol=1e-4)

    def test_gradcheck_above_bound(self):
        """Analytical and numerical gradients agree in the rightmost tile."""
        fta = _make_fta()
        fta._tiling = fta._tiling.double()
        z = torch.tensor(
            [[3.0, 4.0], [5.5, 10.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert gradcheck(fta, (z,), eps=1e-6, atol=1e-4)

    def test_gradcheck_below_bound(self):
        """Analytical and numerical gradients agree below the tiling range."""
        fta = _make_fta()
        fta._tiling = fta._tiling.double()
        z = torch.tensor(
            [[-5.0, -4.0], [-10.0, -3.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        assert gradcheck(fta, (z,), eps=1e-6, atol=1e-4)

    def test_gradient_zero_below_range(self):
        """Gradient is exactly zero for inputs below the tiling range."""
        fta = _make_fta()
        z = torch.tensor([[-5.0]], requires_grad=True)
        out = fta(z)
        out.sum().backward()
        assert torch.all(z.grad == 0)

    def test_gradient_one_in_rightmost_tile(self):
        """Gradient is 1 in the rightmost tile."""
        fta = _make_fta()
        z = torch.tensor([[5.0]], requires_grad=True)
        out = fta(z)
        out.sum().backward()
        assert z.grad.item() == pytest.approx(1.0)
