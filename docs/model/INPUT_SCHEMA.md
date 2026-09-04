# Predictor input schema

Feature semantics must be explicit, typed, unit-bearing, and versioned. Dataframe column positions are not an input contract.

## A. Molecular topology

Atom categorical features describe atom identity and local topology. Bond categorical features describe bond direction, bond type, and ring membership. These remain molecule-level graph inputs.

## B. Molecular geometry and descriptors

Bond length is the continuous bond-level geometry input. Bond angle and molecule-derived descriptors may be encoded on the bond-angle graph when their forward reachability is verified. Experimental conditions must not be represented as if they were bond properties.

## C. Experimental conditions

Clean 4g conditions are graph/sample-level typed inputs:

| Field | Type | Unit/meaning | Construction |
|---|---|---|---|
| `ea_fraction` | `float32`, continuous | unitless fraction in `[0, 1]` | Parse `PE/EA = left/right`; compute `right / (left + right)`. |
| `loading_solvent` | categorical | one of `PE`, `EA`, `DCM` | Vocabulary lookup and embedding; never ordinal numerical semantics. |
| `loading_mass_mg` | `float32`, continuous | mg | `Density g/ml * V/ul`; dimensionally, `1 g/ml * 1 ul = 1 mg`. |
| `loading_solvent_volume_ul` | `float32`, continuous | ul | Direct loading-solvent volume. |

All continuous normalization statistics must be fit only on the current 4g source-train partition. Validation, test, and 8g outcomes or rows contribute zero observations to fitting.

The PE/EA system is binary, so eluent composition has approximately one intrinsic degree of freedom. Six mixture-weighted chemical descriptors may be useful for future multi-solvent generalization, but they are highly collinear in this binary system and are not the Clean 4g model's primary condition input.

## D. Future column context

The Clean 4g predictor does not include column size because it is constant within the source domain and contains no learnable variation. A future multi-column schema should consider physical quantities such as packing mass, column length and diameter, column volume, stationary phase, flow or linear velocity, loading ratio, and other mechanistically meaningful descriptors.

A future multi-column model must not assume that a bare `column=4/8/25/40` code is sufficient to learn physical transfer.
