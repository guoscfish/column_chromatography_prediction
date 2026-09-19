# Selection mechanism summary

Status: **selection-only innovation screening complete**.

This report describes label-free selection geometry only. It does not compare
prediction accuracy, reveal selected labels, or promote a method.

## Mean overlap with raw Gradient-LCMD

| Method | Mean overlap count | Mean overlap fraction |
| --- | ---: | ---: |
| Latent farthest-first | 4.2/32 | 0.131 |
| Latent LCMD | 2.6/32 | 0.081 |
| Gradient farthest-first | 14.4/32 | 0.450 |
| Raw Gradient-LCMD | 32.0/32 | 1.000 |
| Gradient norm Top-B | 5.8/32 | 0.181 |
| Direction LCMD | 0.6/32 | 0.019 |
| Tempered LCMD (alpha=0.5) | 9.2/32 | 0.287 |
| Output-whitened LCMD | 9.8/32 | 0.306 |
| Center/width LCMD | 7.2/32 | 0.225 |

## Representation x selector controls

- Latent -> gradient at fixed farthest-first retained 0.194 of each batch on average.
- Farthest-first -> LCMD at fixed gradient retained 0.450.
- Farthest-first -> LCMD at fixed latent retained 0.406.
- Gradient-norm Top-B retained 0.181 of raw Gradient-LCMD.

These contrasts show how strongly selections change under one controlled
representation or selector substitution. They do not identify a causal
performance mechanism.

## Identical batches

No method pair selected the same ordered membership set in all five seeds.

## Mechanism-only next-stage shortlist

The three most selection-distinct prespecified candidates relative to raw
Gradient-LCMD are: `direction_lcmd`, `latent_lcmd`, `center_width_lcmd`. This is a
diversity-of-geometry shortlist for a separately authorized one-step training
study, not an accuracy ranking.
