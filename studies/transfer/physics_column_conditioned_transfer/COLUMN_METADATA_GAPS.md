# Column metadata gaps

REAL_COLUMN_VOLUME_SCALE_NOT_IDENTIFIABLE_FROM_CURRENT_REPOSITORY_METADATA

Available: raw column_specs, nominal packing masses 4/8/25/40 g, flow, loading quantities, solvent composition and molecule identifiers. 8g is Silica-CS 4g+4g; nominal summed packing mass is 8 g, not evidence of a single cartridge.

Repository clues exist and must not be called absent: application/QGeoGNN.py:1653–1655 hardcodes 8g diameter/length/density 1.5/13.2/0.4458; :1720–1722 hardcodes 2.15/15.6/0.5248 in a legacy dataset path; :3852–3854 hardcodes 4g 1.5/6.6/0.4458. scripts/run_g0_4_paper_style_transfer.py:58–59 repeats 4g/8g tuples. These are implementation constants, without verified physical units, measurement provenance, lot/product IDs, bed-versus-housing definition or void fraction. Do not treat numerical names as verified physical measurements.

Missing verified metadata: measured packed-bed volume, void/dead volume and tracer method; inner diameter and actual bed length; particle-size distribution; measured silica bulk density; manufacturer and exact model/lot; tubing/connection volume for 4g+4g. No internet catalog was substituted. Ask experimental staff to verify code constants against records. No true CV baseline, CV normalization, F/A linear velocity or causal mass/flow interpretation is permitted.

Priorities: crossed mass × flow experiments, independent batch/repeat IDs and exact-condition repeats; high-retention tail and source-unseen compounds. Same raw rows do not establish independent experimental repeats.
