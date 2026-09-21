# Alpha=.8 Fusion-MaxDet truncated developmental screen

This is a two-seed developmental screen through 653 active labels, not an
independent confirmation. Acquisition used the original uncentered latent
representation throughout. No 653-to-1005 extension was run.

## Decision

**WORTH_FULL_TRAJECTORY.** Alpha=.8 remains slightly behind Gradient-MaxDet on
the mean 333-653 curve, but recovers most of the alpha=.5 loss and has a stable
early-stage gain. This satisfies the preregistered "worth a full trajectory"
condition, not the stronger "already matches or exceeds Gradient" condition.

## Primary results

| Seed | Gradient AULC | alpha=.5 AULC | alpha=.8 AULC | alpha=.8 - Gradient AULC | Gradient NRMSE@653 | alpha=.5 NRMSE@653 | alpha=.8 NRMSE@653 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 157 | 0.593462 | 0.605641 | 0.593589 | +0.000127 | 0.484709 | 0.499020 | 0.534840 |
| 6101 | 0.776273 | 0.796518 | 0.785425 | +0.009153 | 0.718619 | 0.733478 | 0.724886 |

Across seeds, alpha=.8 minus Gradient AULC is **+0.004640**. Alpha=.5 minus
Gradient is **+0.016212**, so alpha=.8 recovers about **71%** of the alpha=.5
AULC degradation. Alpha=.8 minus alpha=.5 AULC is **-0.011572**.

The phase result is asymmetric: alpha=.8 minus Gradient is **-0.002795** over
333-525, then **+0.015793** over 525-653. Mean NRMSE@653 remains worse than
Gradient by **+0.028199**. The endpoint degradation is concentrated in seed
157 (+0.050132); seed 6101 is only +0.006267 behind Gradient and is 0.008592
better than alpha=.5.

| Seed | alpha=.8 V1 RMSE | alpha=.8 V1 R2 | alpha=.8 V2 RMSE | alpha=.8 V2 R2 |
|---:|---:|---:|---:|---:|
| 157 | 3.952637 | 0.717762 | 5.923701 | 0.846358 |
| 6101 | 4.410583 | 0.678992 | 7.747584 | 0.717706 |

## Representation and selection

| Seed | Gradient rank / abs corr | latent rank / abs corr | alpha=.5 historical rank / abs corr | alpha=.8 rank / abs corr | centered latent rank / abs corr | r_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 157 | 23.633 / 0.217 | 2.925 / 0.744 | 10.932 / 0.544 | 18.988 / 0.384 | 4.077 / 0.583 | 0.534 |
| 6101 | 24.649 / 0.194 | 2.903 / 0.752 | 9.877 / 0.570 | 19.567 / 0.370 | 4.298 / 0.586 | 0.504 |

Centering roughly raises latent effective rank from 2.91 to 4.19 and lowers
absolute kernel correlation from 0.748 to 0.585. Together with mean-direction
energy ratios near 0.52, this is clear evidence of a strong common-mode
component, although centered latent remains low-rank and correlated.

Mean per-round alpha=.8 batch overlap is **0.211** with Gradient-MaxDet and
**0.256** with alpha=.5 fusion. The high round-0 overlaps rapidly collapse;
therefore a 20% latent kernel weight still materially changes the trajectory.

## Direct answers

1. **Is alpha=.8 more reasonable than alpha=.5?** Yes on the full truncated
   curve: it recovers about 71% of alpha=.5's mean AULC loss, with lower AULC in
   both seeds. Endpoint behavior is mixed rather than uniformly better.
2. **Does it reach or exceed Gradient-MaxDet?** No. Mean AULC is +0.004640
   worse and both seed endpoints are worse, although seed 157's full AULC is
   effectively tied (+0.000127).
3. **Where does improvement occur?** In the early 333-525 phase. The middle
   525-653 phase loses the early advantage.
4. **Does 20% latent still change selection?** Yes. Only 21.1% of selected
   samples overlap Gradient-MaxDet on average, with near-zero late overlap.
5. **Is there common-mode collapse?** Yes. The r_mean values near 0.5 and the
   centered-rank/correlation improvement show a strong common direction, but
   centering does not make latent comparable to Gradient in rank.
6. **Should 653-to-1005 be run?** The preregistered rule says yes: alpha=.8
   recovers most of alpha=.5's loss and has a stable early-stage gain. This is a
   developmental recommendation only; the extension has not been started.

All 22 alpha=.8 prediction points and 20 acquisition batches were frozen before
the one-time test reveal. No statistical-significance claim is made from two
seeds.
