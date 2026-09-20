# ResFlow architecture notes

This repository is an unofficial implementation based only on Figure 7 and
the public paper/supplement. It does not claim to reproduce unpublished author
code.

## Specified by ResFlow paper

- The velocity network uses a DDPM-style U-Net backbone. Its primary state
  input is `x_t`, and timestep `t` conditions the U-Net blocks through
  adaptive normalization rather than an additive timestep bias.
- The network predicts the complete augmented velocity
  `v_theta(x_t, y_t, t) = [v_x, v_y]`. For RGB data the output therefore has
  six channels.
- `y_t` is not concatenated with, or added to, `x_t` at the U-Net input. It is
  processed by an independent Adapter.
- The Adapter is a serial stack of residual blocks. Later stages progressively
  downsample the preceding state to produce spatially aligned multi-scale
  features.
- Adapter features modulate corresponding U-Net features using AdaLN-style
  adaptive normalization rather than addition or concatenation.
- The output projection of every Adapter residual stage is zero-initialized.
  Consequently all Adapter features presented to the U-Net are exactly zero at
  initialization, and the auxiliary branch initially has no effect on the
  timestep-conditioned backbone.
- Following the encoder interpretation of Figure 7, the four Adapter outputs
  are injected at four encoder/down-path resolution stages. For the 64x64
  configuration with the default DDPM channel schedule, the mapping is:

  | Adapter output | Shape | Encoder target |
  |---|---|---|
  | `a0` | `[B, 128, 64, 64]` | encoder level 0, 64x64 |
  | `a1` | `[B, 128, 32, 32]` | encoder level 1, 32x32 |
  | `a2` | `[B, 256, 16, 16]` | encoder level 2, 16x16 |
  | `a3` | `[B, 256, 8, 8]` | encoder level 3, 8x8 |

## Not specified by paper / implementation choices

- **Normalization type.** The publication says AdaLN but does not provide the
  convolutional normalization implementation. `AdaptiveNorm` uses GroupNorm
  without affine parameters by default. This is an AdaLN-style implementation,
  not a claim that the private code used GroupNorm.
- **Condition fusion formula.** Timestep and auxiliary parameters are projected
  independently and combined as

  ```text
  (1 + gamma_t + gamma_y) * Norm(h) + beta_t + beta_y
  ```

  This formula is an explicit implementation choice. Blocks without an
  auxiliary injection use only `gamma_t` and `beta_t`.
- **Injection location within a stage.** There is one auxiliary injection per
  encoder scale, placed in the adaptive normalization of the final residual
  block in each of encoder levels 0--3. The paper does not disclose the exact
  residual-block index. Decoder levels, the 4x4/2x2 encoder levels, and the
  bottleneck receive timestep conditioning but no Adapter feature.
- **Timestep scale.** Raw continuous `t` in `[0, 1]` is used by default
  (`timestep_scale=1.0`). Whether the authors used `t`, `1000*t`, or another
  scaling is not public.
- **Exact DDPM topology.** The defaults use base width 128, channel multipliers
  `[1, 1, 2, 2, 4, 4]`, two residual blocks per encoder level, three per
  decoder level, and attention at 16x16. The paper does not publish a complete
  layer table, channel table, head count, or attention placement.
- **Adapter internals.** Each stage uses two 3x3 convolutions plus a residual
  skip, followed by a zero-initialized 1x1 output projection. Kernel sizes,
  activation, channel counts, and the exact downsampling operator are not given
  by the paper.
- **Four-stage interpretation.** Figure 7 depicts four serial Adapter residual
  stages and four encoder AdaNorm injections. The public text does not state
  whether that drawing is schematic or whether a private implementation also
  conditions deeper U-Net resolutions.
- **Checkpoint compatibility.** Checkpoints produced by the earlier simplified
  model are intentionally incompatible. Timestep projections changed from
  `C`-channel additive biases to `2C` scale/shift projections; adaptive norms
  changed parameterization; the Adapter changed from six outputs to four; and
  eighteen decoder modulation modules were replaced by four encoder injection
  points. The final RGB velocity head remains six-channel, but that alone does
  not make old state dictionaries loadable.
