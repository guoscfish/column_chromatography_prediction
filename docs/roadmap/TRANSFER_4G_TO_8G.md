# 4g to 8g transfer roadmap

Status: `PAUSED_PENDING_PREDICTOR_QUALIFICATION`.

All routes below are `FUTURE / NOT AUTHORIZED`. The current data should first be described as 4g source and 8g target domains; they are not automatically a rigorous low-/high-fidelity pair.

## Route A: transfer adaptation

Qualify a Clean 4g source model, then adapt it with limited 8g labels under a new frozen protocol. This is the most immediately realistic route, but it must not begin before formal 4g predictor qualification.

## Route B: multi-column joint modeling

If sufficiently comparable 4g, 8g, 25g, and 40g observations become available, jointly model explicit column context. Prefer measured physical quantities: packing mass, column length, diameter, volume, stationary phase, flow or linear velocity, loading ratio, and other mechanistically meaningful descriptors.

A categorical or numeric `column=4/8/25/40` code is not presumed to be a sufficient physical description.

## Route C: shared representation with domain correction

Use a shared molecular/condition representation with a column- or domain-specific residual, adapter, or correction. Multi-domain and multi-fidelity literature may inform the architecture, but analogy does not establish that current column domains are fidelity levels.

No transfer training, target-label selection, or 8g performance comparison is authorized by this roadmap.
