# ResFlow auxiliary-variable intervention report

Checkpoint: `best_step_0030000_psnr_31.0606.pt` (step 30,000)

Protocol: four uniform Euler steps (`1 -> .75 -> .5 -> .25 -> 0`), fixed seed
`20250920`, 322 paired test images. PSNR/SSIM are RGB with no border crop,
LPIPS uses AlexNet, and FID uses torch-fidelity Inception-v3 2048-D features.
`FID(all)` uses all 8,000 train pairs plus 322 test pairs.

## Baseline metrics

| PSNR | SSIM | LPIPS | FID(test) | FID(all) |
|---:|---:|---:|---:|---:|
| 31.06094 | 0.912226 | 0.0104983 | 13.29276 | 1.32159 |

## Trajectory interventions

| Mode | PSNR | SSIM | LPIPS | LPIPS vs persistent | FID(test) |
|---|---:|---:|---:|---:|---:|
| Persistent | 31.06094 | 0.912226 | 0.0104983 | 0 | 13.29276 |
| Step-resample | 31.06126 | 0.912240 | 0.0104979 | 0.00000188 | 13.29776 |
| Fixed-typical | 31.06177 | 0.912232 | 0.0104976 | 0.00000533 | 13.27583 |
| Swap after first step | 31.06093 | 0.912238 | 0.0104948 | 0.00000267 | 13.30192 |
| Swap after second step | 31.06093 | 0.912239 | 0.0104966 | 0.00000134 | 13.31313 |
| Swap after third step | 31.06091 | 0.912234 | 0.0104976 | 0.00000051 | 13.30166 |

Persistent and step-resample are statistically indistinguishable at the reported
precision. Abrupt identity swaps at every possible transition are also negligible.

## Multi-y output diversity

Measured on 32 test inputs with 20 independent persistent identities per input:

- mean per-pixel variance on `[0,1]`: `1.8271e-7`
- mean pairwise output LPIPS: `6.5743e-6`
- DCT variance energy: low `38.53%`, mid `33.45%`, high `28.02%`
- mean DCT coefficient variance: low `7.2558e-7`, mid `2.1040e-7`, high `7.9510e-8`

The already tiny variation is not concentrated in the high-frequency band; the
low-frequency coefficient variance is the largest.

## Direct y -> velocity dependence at fixed x_t

Measured on 32 persistent rollout states, with 20 independent `y_t` samples while
holding each `x_t` fixed:

| t | velocity variance | variance / velocity energy | finite-difference sensitivity |
|---:|---:|---:|---:|
| 1.00 | 6.8584e-7 | 4.6006e-5 | 0.005498 |
| 0.75 | 5.7035e-7 | 4.0855e-5 | 0.005178 |
| 0.50 | 6.0238e-7 | 4.3619e-5 | 0.005305 |
| 0.25 | 8.0637e-7 | 5.9204e-5 | 0.005850 |

The dependence is nonzero but extremely small and does not decay toward the HQ
endpoint; it is largest at `t=.25`, opposite to the expected uncertainty-scope
trend.

## Conclusion

For this checkpoint and dataset, the auxiliary input does not behave as a
trajectory-level branch identity. Destroying cross-time identity does not reduce
quality, different persistent identities produce almost identical outputs, and
the direct effect on `v_x` is about `4e-5` to `6e-5` of velocity energy. At most,
`y_t` acts as negligible stochastic feature modulation; the experiments do not
support the claimed branch-selection interpretation.
