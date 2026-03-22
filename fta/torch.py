"""PyTorch implementation of the Fuzzy Tiling Activation (FTA).

Based on the Fuzzy Tiling Activation from Pan, Banman & White (2021),
extended with a right tile that linearly transforms inputs above the
core tiling range (in effect, outside the core tiling range, this
implementation behaves as a shifted ReLU).

Reference: https://arxiv.org/abs/1911.08068
"""

import torch


class FTA(torch.nn.Module):
    """Fuzzy Tiling Activation with an extra linear tile.

    Tiles the interval ``[-bound, bound]`` and produces a fuzzy, sparse
    encoding for each input feature.  Inputs above the tiling range are
    passed through a shifted linear tile; inputs below the tiling range
    are zeroed out.

    The output dimensionality per feature is ``num_tiles + 1`` (the extra
    tile is the linear one).

    Parameters
    ----------
    bound : float
        Upper bound of the tiling range; the lower bound is ``-bound``.
    spillover_base : float
        Base value used to derive the spillover (called "sparisty parameter"
        and denoted as *eta* in the paper). Higher spillover means lower
        sparsity.
    spillover_mode : str
        How ``spillover_base`` is converted to the actual spillover:

        * ``"derive_from_bound"``: ``bound / 2 ** spillover_base``
        * ``"derive_from_tile_width"``: ``tile_width * 1.5 ** spillover_base``
        * ``"raw"`` or None: use ``spillover_base`` directly.
    tile_width : float or None
        Width of each tile.  Provide either ``tile_width`` or
        ``num_tiles`` (if both are given, ``tile_width`` is used).
    num_tiles : int or None
        Number of tiles in the tiling.  Provide either ``num_tiles`` or
        ``tile_width`` (if both are given, ``tile_width`` is used).
    device : torch.device, optional
        Device for internal tensors (default: CPU).
    """

    def __init__(
        self,
        bound,
        spillover_base,
        spillover_mode,
        tile_width,
        num_tiles,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()

        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if not (tile_width or num_tiles):
            raise ValueError("must provide one of tile_width or num_tiles")
        if num_tiles is not None and num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
        if tile_width is not None and tile_width <= 0:
            raise ValueError(f"tile_width must be > 0, got {tile_width}")

        self._tile_width = tile_width or 2 * bound / num_tiles
        self._tiling = torch.arange(
            -bound,
            bound,
            self._tile_width,
            device=device
        )

        # Keep _num_tiles as the number of core tiles; the final activation
        # will have an extra tile
        self._num_tiles = len(self._tiling)

        if spillover_mode == 'derive_from_bound':
            self._spillover = bound / 2**spillover_base
        elif spillover_mode == 'derive_from_tile_width':
            self._spillover = self._tile_width * 1.5**spillover_base
        elif spillover_mode == "raw" or spillover_mode is None:
            if spillover_base < 0:
                raise ValueError(
                    f"spillover_base must be >= 0 when "
                    f"spillover_mode is 'raw' or None, "
                    f"got {spillover_base}"
                )
            self._spillover = spillover_base
        else:
            raise ValueError(
                f"spillover_mode must be 'derive_from_bound', "
                f"'derive_from_tile_width', 'raw', or None, "
                f"got {spillover_mode!r}"
            )

    def forward(self, z):
        """Apply the activation.

        Parameters
        ----------
        z : Tensor
            Input tensor of shape ``(batch, features)``.

        Returns
        -------
        Tensor
            Shape ``(batch, (num_tiles + 1) * features)``.
        """
        return _FTAFunction.apply(
            z,
            self._tiling,
            self._tile_width,
            self._spillover
        )

    @property
    def num_tiles(self):
        """Total number of tiles (including the rightmost linear one)"""
        return self._num_tiles + 1

    @property
    def spillover(self):
        return self._spillover

    @property
    def tile_width(self):
        return self._tile_width


class _FTAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, tiling, tile_width, spillover):
        ctx.save_for_backward(z)

        ctx.tiling = tiling
        ctx.tile_width = tile_width
        ctx.spillover = spillover

        device = z.device

        # Compute the core fuzzy tiling activation
        core_output = torch.tensor(1.0, device=device) - _fuzzy_indicator(
            _intermediate(z, tiling, tile_width),
            spillover
        )  # [batch, num_tiles, features]

        # Determine the tiling's upper bound
        # Note: tiling was constructed as
        # torch.arange(-bound, bound, tile_width), so the upper bound is
        # tiling[-1] + tile_width
        upper_bound = tiling[-1] + tile_width
        ctx.upper_bound = upper_bound

        # For inputs below the upper bound, use the computed tiles;
        # otherwise, we set the core tiles to zero
        inside_mask = (z <= upper_bound).to(core_output.dtype)
        core_output = core_output * inside_mask.unsqueeze(1)

        # Compute the rightmost linear tile
        right_tile = torch.where(
            z > upper_bound,
            z - upper_bound,
            torch.zeros_like(z, device=device)
        ).unsqueeze(1)

        # Concatenate the rightmost tile to the core tiles
        final_output = torch.cat(
            [core_output, right_tile],
            dim=1
        )  # [batch, num_tiles + 1, features]

        # Flatten the output
        return final_output.view(
            final_output.shape[0],
            -1
        )  # [batch, (num_tiles + 1) * features]

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors

        tiling = ctx.tiling
        tile_width = ctx.tile_width
        spillover = ctx.spillover
        upper_bound = ctx.upper_bound

        # Compute the Jacobian for the core fuzzy tiling tiles
        jacobian = _jacobian(z, tiling, tile_width, spillover)

        num_core_tiles = tiling.shape[0]
        num_total_tiles = num_core_tiles + 1  # core tiles plus right tile

        # Unflatten grad_output
        unflattened_grad_output = grad_output.view(
            grad_output.shape[0],
            num_total_tiles,
            -1
        )  # [batch, num_total_tiles, features]

        # Split the gradients into two parts
        mid_grad = unflattened_grad_output[:, 0:-1, :]  # from the core tiles
        right_grad = unflattened_grad_output[:, -1, :]  # from the right tile

        # Only propagate gradients from the core tiles if z is inside the
        # tiling range
        inside_mask = (z <= upper_bound).to(z.dtype)
        mid_grad_contrib = torch.sum(jacobian * mid_grad, dim=1) * inside_mask

        # For the extra tile, the function is simply y=x, so the derivative is
        # 1 when active
        right_mask = (z > upper_bound).to(z.dtype)
        right_grad_contrib = right_grad * right_mask

        # Total gradient is the sum of contributions from the two parts
        grad_input = mid_grad_contrib + right_grad_contrib

        return grad_input, None, None, None


def _jacobian(z, tiling, tile_width, spillover):
    device = z.device

    dhdz = torch.zeros(
        z.shape[0],
        tiling.shape[0],
        z.shape[1],
        device=device
    )

    left_borders = (
        tiling[0] +
        tile_width * torch.arange(tiling.shape[0], device=device)
    )
    right_borders = (
        tiling[0]
        + tile_width * (torch.arange(tiling.shape[0], device=device) + 1)
    )

    cond1 = (tiling[0] <= z) & (z <= tiling[-1] + tile_width)
    cond2 = (
        (left_borders.view(1, -1, 1) - spillover < z.unsqueeze(1))
        & (z.unsqueeze(1) < left_borders.view(1, -1, 1))
    )
    cond3 = (
        (right_borders.view(1, -1, 1) < z.unsqueeze(1))
        & (z.unsqueeze(1) < right_borders.view(1, -1, 1) + spillover)
    )

    dhdz[cond1.unsqueeze(1) & cond2] = 1
    dhdz[cond1.unsqueeze(1) & cond3] = -1

    return dhdz


def _intermediate(z, tiling, tile_width):
    device = z.device
    return (
        torch.maximum(
            tiling.view(1, -1, 1) - z.unsqueeze(1),
            torch.tensor(0.0, device=device)
        ) +
        torch.maximum(
            z.unsqueeze(1) - tile_width - tiling.view(1, -1, 1),
            torch.tensor(0.0, device=device)
        )
    )


def _indicator(x):
    device = x.device
    threshold = 1e-6  # to handle floating point precision issues
    return (x > torch.tensor(threshold, device=device)).to(int)


def _fuzzy_indicator(x, spillover):
    device = x.device
    return (
        (torch.tensor(1.0, device=device) - _indicator(x - spillover)) * x
        + _indicator(x - spillover)
    )
