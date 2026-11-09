# Candidate retirement proof (2026-11-09)

Only the isolated candidate removes two dated plot entry points. Active main,
all frozen code, results, source tables, and live controllers retain their paths
and bytes. Gate 3 remains stopped.

| Retired entry | New authority | Preserved behavior |
|---|---|---|
| `scripts/studies/plot_free_cw_snapshot_20261102.py` | `scripts/plot_saved_validation.py`, kind `validation-snapshot` | Five-method common-budget mean curves, current per-seed completed-fit curves, REPORT link and snapshot-manifest updates |
| `scripts/studies/plot_free_rerun_audit_20261103.py` | Same CLI, kind `rerun-audit` | Four original evidence panels, annotations, font and axis settings, checkpoint byte-hash check |

The rendering function bodies were moved into a shared config-driven entry point.
The dated files are deleted, rather than replaced with forwarding modules. The new
entry point never imports either old script. Historical report prose and the old
`plot_script` manifest field are preserved as citations to the original commit;
the replacement mapping and restore command live in `RETIREMENT_MANIFEST.json`.

The source-only dependency audit found no executable importer, direct subprocess
invocation, frozen protocol hash, protected artifact, or live-process binding for
either retired entry. The remaining Python references are the synthetic regression
which reads original source through `git show`, and the integrity verifier's approved
retirement list. These are reviewed individually as nonruntime references in the
manifest. Historical report/manifest citations are not rewritten. Broad string scans
remain conservative; they do not prove arbitrary external Python reflection absent.

`tests/refactor/test_plot_saved_validation.py` constructs only synthetic aggregate
validation curves, synthetic fit audit metadata and fake checkpoint bytes. With
fixed `SOURCE_DATE_EPOCH` and SVG hash salt, old and new PNG, SVG and PDF files are
byte-identical; the snapshot REPORT and manifest are also byte-identical. Exact
hashes are recorded in `PLOT_EQUIVALENCE.json`. No actual historical result, test
truth, provider payload, live checkpoint or live log is consumed by this regression.
The fixed rendering metadata is test-only and does not alter historical reports.

Example config schema is in `configs/plots/*.example.json`:

```sh
python scripts/plot_saved_validation.py --config path/to/authorized_plot_config.json
```

Run only against an independent authorized snapshot; the examples intentionally do
not point to an active experiment. This task did not run either new rendering mode
on real experiment data.
