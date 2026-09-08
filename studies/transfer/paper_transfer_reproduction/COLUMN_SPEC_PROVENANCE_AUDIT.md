# Column specification provenance audit

## Observed repository constants

| Column | Values in released code | Locations |
| --- | --- | --- |
| 4g | `(1.5, 6.6, 0.4458)` | `application/QGeoGNN.py:3851-3855`; repeated in G0-4 |
| 8g | `(1.5, 13.2, 0.4458)` | `application/QGeoGNN.py:1652-1656`; `scripts/run_g0_4_paper_style_transfer.py:58-59` |
| 25g | `(2.15, 15.6, 0.5248)` | `application/QGeoGNN.py:1719-1723` |
| 40g | `(2.15, 15.6, 0.5248)` | `application/QGeoGNN.py:1786-1790` |

The optional feature names are explicitly `column_dia`, `column_len`, and
`column_den` at `application/QGeoGNN.py:945-946` and in the RBF encoder registry at
`1027-1029`. The ordering above follows that code.

## Provenance conclusion

The 25g and 40g tuples are repeated in their respective dataset constructors and no
other repository annotation, README, commit history, product identifier, unit
definition, experimental log, or original checkpoint establishes their physical
provenance. The source data labels identify distinct column specifications,
`Silica-CS 25g` and `Silica-CS 40g`, while the model-input tuple is identical.

This reproduction therefore uses the tuple as
`repository-original hardcoded values`, not as verified geometry or material
properties. Every primary result carries
`25G_AND_40G_SHARE_LEGACY_COLUMN_SPEC_VALUES`. It must not be interpreted as
evidence that the two physical columns have the same diameter, bed length, density,
or void volume. The original values are retained because changing a suspicious
constant would cease to be a faithful code reproduction.
