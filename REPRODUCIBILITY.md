# Reproducibility audit

This repository follows the CVPR 2025 paper and its supplementary material. It is unofficial: the [authors' repository](https://github.com/luowyang/ResFlow) still contains only a “code is coming soon” notice at the time this implementation was prepared.

## Directly specified by the paper

| Item | Implemented value | Source |
|---|---:|---|
| Image path | `x_t=(1-t)x_0+t x_1` | Eq. (7) |
| Auxiliary path | `sigma_y=beta/(1-t+beta)` | Eq. (8), Appendix Eq. (22) and Fig. 6 |
| Schedule parameter | `beta=10` | Section 3.2 / Appendix B |
| Loss | weighted L2 velocity matching for both `x` and `y` | Eqs. (9)-(10) |
| Loss weighting | `(cos(pi/2*(t-2))+1)^gamma` | Eq. (11) |
| Weight exponent | `gamma=1.75` | Section 3.3 |
| Backbone | DDPM U-Net for every task | Section 4.1 / Appendix C |
| Auxiliary condition | residual adapter, downsampled features, adaptive normalization; zero-initialized adapter outputs | Appendix C, Fig. 7 |
| Input normalization | `[-1,1]` | Appendix C |
| Train crop | random `256x256` | Section 4.1 / Appendix C |
| Test size | full resolution | Section 4.1 |
| Optimizer | AdamW, betas `(0.9,0.999)` | Appendix C |
| Learning rate | cosine `1e-4` to `1e-6` | Appendix C |
| Training length | 400,000 iterations | Appendix C |
| Hardware | 8 NVIDIA A100 GPUs | Appendix C |
| Inference | uniform-time, four-step Euler | Sections 3.3, 4.1, Appendix D |

## Underspecified or inconsistent items

These cannot be reproduced *strictly* without clarification or official code. They are exposed in `configs/_base.yaml`, rather than hidden in the implementation.

1. **Exact U-Net dimensions are absent.** “The same U-Net architecture as DDPM” does not identify which DDPM experiment. The default uses Ho et al.'s official 256×256 LSUN setting: width 128, multipliers `[1,1,2,2,4,4]`, two residual blocks, attention at 16, dropout 0. This is a documented inference, not a ResFlow-paper value.
2. **Batch-size scope is absent.** The supplement says batch size 8 and eight A100s, but not whether 8 is global or per GPU. The default interprets it as global batch 8 (one sample/GPU), which is also plausible for the 114M-parameter 256 U-Net. Change `train.global_batch_size` if author clarification says otherwise.
3. **AdamW weight decay is absent.** The default is PyTorch's `0.01`; override it explicitly for comparisons. The main paper says Adam while the supplement says AdamW; the more detailed supplement is followed.
4. **Precision, EMA, gradient clipping, seed, and augmentation beyond cropping are absent.** Defaults are FP32, no EMA, no clipping, seed 0, and no flip. These avoid silently adding unreported training techniques.
5. **The auxiliary endpoint is internally inconsistent.** The prose and Eq. (3)/(5) say `y_0=0`, but the displayed/figured final schedule gives `sigma_y(0)=beta/(1+beta)` (10/11 for beta 10), not zero. Fig. 6 confirms the nonzero value. The code follows the displayed equation and figure literally. At inference it analytically replaces `y_t` from the sampled `y_1`, as required by Section 3.3, and discards predicted auxiliary state updates.
6. **Adaptive-normalization mechanics and velocity channel count are not fully specified.** Equations (6), (9), and (10) require predicting concatenated `(v_x,v_y)`, so the implementation outputs six channels and trains both. Following the encoder interpretation of Fig. 7, four serial zero-output Adapter stages condition four encoder scales through the documented AdaLN-style module. GroupNorm and the exact joint time/auxiliary scale-shift formula are implementation choices documented in `ARCHITECTURE_NOTES.md`.
7. **Metric details are absent.** The paper does not state RGB versus Y-channel PSNR/SSIM, border shaving, SSIM padding, LPIPS backbone/version, or stochastic-sample aggregation. Evaluation here uses RGB `[0,1]`, no border crop, Wang SSIM with an 11×11 Gaussian valid window, and optional AlexNet LPIPS.
8. **Some dataset splits/preprocessing are ambiguous.** Configs mirror the counts and train/test roles reported by the paper, but do not redistribute datasets. JPEG uses fixed quality 10 with 4:4:4 PIL encoding; the paper reports QF=10 but not codec/library or chroma subsampling.

These limitations mean matching the reported numbers exactly cannot be guaranteed from the publication alone. The repository is designed so that author clarifications can be applied by changing YAML values or a small, isolated module.
