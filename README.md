# FTA - Fuzzy Tiling Activation

FTA is a neural network activation function that converts each scalar input into a tiled vector with "soft" or "fuzzy" sparsity: between the fully active and fully inactive tiles are one or more partially active tiles. FTA has been shown to be robust and effective in a wide variety of deep reinforcement learning settings (including continual learning), typically beating ReLU and tanh ([Pan, Banman & White (2021)](https://arxiv.org/abs/1911.08068); [Lazar (2025)](https://ualberta.scholaris.ca/items/186a1735-37cb-408e-85a2-01a19f4a96ce); Lazar, Vandergrift, White, & White (forthcoming, 2026)).

This implementation extends the original from [Pan, Banman & White (2021)](https://arxiv.org/abs/1911.08068) by adding an extra tile at the right end of the core tiling. This tile implements a ReLU-like function that is shifted so that the discontinuity (where the flat section and linear section meet) lands on the core tiling's right boundary.

Both **PyTorch** and **JAX/Flax** implementations are included.


## Installation

```bash
# PyTorch only
pip install fuzzy-tiling-activation[torch]

# JAX / Flax only
pip install fuzzy-tiling-activation[jax]

# Both
pip install fuzzy-tiling-activation[torch,jax]
```

## Quick start

FTA is used in the same way as any built-in activation function. Note that because it's one-to-many rather than one-to-one, it increases the output dimensionality of its layer by a factor of `num_tiles + 1`.


### PyTorch

```python
from fta.torch import FTA

activation = FTA(
    bound=4.0,
    spillover_base=2,
    spillover_mode="derive_from_bound",
    num_tiles=8,
)

# z has shape (batch, features)
out = activation(z)  # shape: (batch, (num_tiles + 1) * features)
```

### JAX / Flax

```python
from fta.jax import FTA

activation = FTA(
    bound=4.0,
    spillover_base=2,
    spillover_mode="derive_from_bound",
    num_tiles=8,
)

params = activation.init(rng_key, z)
out = activation.apply(params, z)
```


## Parameters

| Parameter | Description |
|---|---|
| `bound` | Upper bound of the tiling range (lower bound is `-bound`) |
| `spillover_base` | Controls sparsity — higher means less sparse |
| `spillover_mode` | How `spillover_base` is converted to the actual spillover: `"derive_from_bound"` uses `bound / 2^spillover_base`; `"derive_from_tile_width"` uses `tile_width * 1.5^spillover_base`; `"raw"` (or `None`) uses `spillover_base` directly |
| `tile_width` | Width of each tile (provide this *or* `num_tiles`) |
| `num_tiles` | Number of tiles (provide this *or* `tile_width`) |


## License

MIT
