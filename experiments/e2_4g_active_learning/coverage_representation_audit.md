# Coverage Representation Audit

The current Coverage and Hybrid diversity stage uses `z = [h_graph; conditions]`.

- `h_graph` is the 128-dimensional graph representation returned by the frozen QGeoGNN prediction interface.
- The 9 condition dimensions are six eluent-descriptor dimensions from `PE/EA`, followed by encoded loading solvent, loading mass (`Density g/ml × V/ul`), and loading-solvent volume.
- Graph and condition dimensions are independently feature-wise z-scored using only the current non-validation labeled training set. Pool and test rows never fit normalization statistics.
- The standardizer is refit each active-learning round because the current labeled training reference changes; validation remains excluded.
- The two standardized blocks are concatenated and Euclidean distance is used.
- Farthest-first starts with the pool candidate farthest from the labeled reference and updates every remaining candidate's minimum distance after each selected batch element. Ties use lexicographic `sample_id` order.

## Open Representation Risk

Feature-wise z-scoring does not balance blocks. The graph block has 128 dimensions and the condition block has 9, so Euclidean distance may still be dominated by graph representation. The E2 row protocol is not changed post hoc.

Future E3 representation ablation should compare `h_graph only`, `conditions only`, current `h_graph + conditions`, block-balanced `h_graph + conditions`, and, if needed, MorganFP + conditions. None of these ablations is run in this stage.
