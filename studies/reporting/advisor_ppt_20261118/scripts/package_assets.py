"""Index figure evidence, merge metrics and package the reviewed deliverables."""
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
from zipfile import ZIP_DEFLATED, ZipFile

REPORT = Path(__file__).resolve().parents[1]
REPO = REPORT.parents[2]
SOURCES = REPORT / "sources"
ACTIVE = Path("studies/active_learning")
LLM = ACTIVE / "llm_decision_behavior_audit_20261116"
LONG = (ACTIVE / "nocw_vs_cw_lcmd_five_to1485_20261112"
        / "reports/final_five_seed_L1485_20261116")
HISTORY = ACTIVE / "comprehensive_history_diagnostics_20261112"
MECHANISM = ACTIVE / "decision_mechanism_followup_20261117"
RETRY = ACTIVE / "volume_response_retry_audit_20261117"
BRANCH = ACTIVE / "qgeognn_v2_same_state_branching"


def variants(path, extensions):
    return [path.with_suffix("." + extension) for extension in extensions]


ORIGINALS = {
    "F01": (variants(LLM / "FIGURES/figure5_row_performance", ["svg", "pdf", "png"])
            + variants(LLM / "FIGURES/figure7_llm_trajectories", ["svg", "pdf", "png"])
            + variants(LONG / "five_seed_raw_learning_curves", ["pdf", "png"])),
    "F02": variants(LLM / "FIGURES/figure5_row_performance", ["svg", "pdf", "png"]),
    "F03": variants(MECHANISM / "figures/hybrid_phase_differences", ["pdf", "png"]),
    "F04": variants(MECHANISM / "figures/hybrid_counterfactual_tradeoff", ["pdf", "png"]),
    "F05": (variants(RETRY / "audit_figures/c2_first_failure_and_prefix", ["pdf", "png"])
            + variants(RETRY / "audit_figures/c4_local_repair_preservation_and_gaps", ["pdf", "png"])),
    "F06": variants(RETRY / "audit_figures/a2_frozen_condition_amplitudes", ["pdf", "png"]),
    "F07": (variants(RETRY / "figures/controls_paired_seeds", ["pdf", "png"])
            + [RETRY / "figures/controls_response_amplitude.png"]),
    "F08": [HISTORY / "figures/learning_all_methods.png"],
    "F09": [HISTORY / "figures/model_quantile_reliability.png"],
}

REFERENCES = {
    "F01": [LLM / "REPORT.md", LLM / "PROTOCOL_AUDIT.md", LLM / "VISUAL_REVIEW.md",
            LONG / "artifact_hashes.json"],
    "F02": [LLM / "performance_summary.csv", LONG / "paired_results.csv",
            HISTORY / "PROTOCOL_AND_FAIRNESS_AUDIT.md"],
    "F03": [MECHANISM / "README.md", MECHANISM / "PHASE_PERFORMANCE.md",
            MECHANISM / "phase_overall.csv", LLM / "PROTOCOL_AUDIT.md"],
    "F04": [MECHANISM / "HYBRID_COUNTERFACTUAL.md", MECHANISM / "counterfactual_input_manifest.json"],
    "F05": [RETRY / f"{name}.md" for name in
            ["FINAL_DECISION", "PREFIX_FAILURE_ANALYSIS", "LOCAL_REPAIR_DESIGN", "AUDIT_FIGURES"]]
           + [MECHANISM / "LLM_REPRODUCIBILITY.md"],
    "F06": [MECHANISM / "MODEL_LEARNABILITY.md", RETRY / "AUDIT_FIGURES.md"],
    "F07": [RETRY / "ANALYSIS_PLAN.json", RETRY / "MODEL_COMPARISON.md",
            RETRY / "FINAL_DECISION.md", RETRY / "RESULTS_INDEPENDENT_QA.json"],
    "F08": [HISTORY / f"{name}.md" for name in
            ["MASTER_REPORT", "PROTOCOL_AND_FAIRNESS_AUDIT", "RESEARCH_DIRECTIONS"]],
    "F09": [HISTORY / "NUMERICAL_SIGNAL_AUDIT.md", HISTORY / "tables/model_seed_phase_errors.csv"],
    "P12": [BRANCH / "FINAL_REPORT.md",
            ACTIVE / "gollum_adaptation_fusion_20261106/NEXT_STAGE_DECISION.md",
            ACTIVE / "representation_gp_benchmark_20261104/NEXT_STAGE_DECISION.md",
            ACTIVE / "gp_guided_llm_stability_20261103/NEXT_STAGE_DECISION.md",
            ACTIVE / "epig_prediction_oriented_20261117/NEXT_STAGE_DECISION.md"],
}


def digest(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_metrics():
    fragments = sorted(SOURCES.glob("*_metrics_fragment.csv"))
    assert len(fragments) == 4
    columns, rows = load_csv(fragments[0])
    for fragment in fragments[1:]:
        new_columns, new_rows = load_csv(fragment)
        assert new_columns == columns, fragment
        rows.extend(new_rows)
    assert len(rows) > 500
    rows.sort(key=lambda row: row["figure"])
    write_csv(REPORT / "metrics_source.csv", columns, rows)
    return rows


def index_sources(metrics):
    evidence = {(figure, path): "原报告图"
                for figure, paths in ORIGINALS.items() for path in paths}
    evidence.update({(figure, path): "原报告或交叉核验"
                     for figure, paths in REFERENCES.items() for path in paths})
    for row in metrics:
        for path in re.split(r"\s+\|\s+|;\s+", row["source"]):
            relative = Path(path)
            assert not relative.is_absolute() and ".." not in relative.parts
            for figure in row["figure"].split("/"):
                evidence.setdefault((figure, relative), "直接复算或归档计数")
    hashes = {path: digest(REPO / path) for _, path in evidence}
    output = [
        {"figure_or_page": figure, "role": role, "repo_path": path.as_posix(), "sha256": hashes[path]}
        for (figure, path), role in sorted(evidence.items())
    ]
    write_csv(SOURCES / "figure_sources.csv", ["figure_or_page", "role", "repo_path", "sha256"], output)


def package():
    members = [*REPORT.glob("*.md"), REPORT / "metrics_source.csv"]
    for folder in ("plots", "rendered_previews", "scripts", "sources"):
        members.extend(item for item in (REPORT / folder).iterdir() if item.is_file())
    members = sorted(members)
    compared = json.loads((SOURCES / "visual_qa.json").read_text())
    assert len(compared) == 9
    for figure in compared:
        for extension, checksum in figure["sha256"].items():
            assert digest(REPORT / "plots" / f"{figure['stem']}.{extension}") == checksum
        assert digest(REPORT / "rendered_previews" / f"{figure['stem']}.png") == figure["pdf_proof_sha256"]
    destination = REPORT / "PPT_ASSETS.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=8) as archive:
        for member in members:
            archive.write(member, member.relative_to(REPORT).as_posix())
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [path.relative_to(REPORT).as_posix() for path in members]
    print(f"{len(members)} archived members; ZIP verified; SHA256 {digest(destination)}")


if __name__ == "__main__":
    index_sources(merge_metrics())
    package()
