# Row Gradient-MaxDet B32 study

This development study compares two new acquisition policies over the complete
333-to-1005 row-split learning curve:

- `gradient_maxdet`: greedy conditional D-optimal design on the authoritative
  512-dimensional full-network q50 gradient CountSketch;
- `u50_gradient_maxdet`: the identical selector restricted to the fixed top 50%
  of current-pool K=3 ensemble disagreement.

The study reuses exactly matching historical round-zero artifacts and all
published Random, Gradient-LCMD, Hybrid, Kernel-IVR, and full-data reference
results. Later Hybrid ensemble artifacts are forbidden because the U50-MaxDet
trajectory has a different labeled state.

The executed cohort is intentionally reduced to two seeds (`157`, `6101`) to
control runtime. This is a developmental subset of the historical five-seed
cohort, not an independent confirmation. All executed trajectories are frozen
behind one global pre-test barrier.
