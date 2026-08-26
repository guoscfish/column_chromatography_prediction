from pathlib import Path
import re

ROOT = Path(__file__).parents[1]

def test_index_keys_unique_and_d45_registered():
    text = (ROOT / "experiments/INDEX.md").read_text()
    keys = re.findall(r"^\| ([^|]+) \|", text, re.MULTILINE)
    assert len(keys) == len(set(keys))
    assert keys.count("d45_oracle_marginal_utility") == 1

def test_d45_script_registered():
    assert "run_d45_oracle_marginal_utility.py" in (ROOT / "scripts/README.md").read_text()


def test_d46_registered_once_and_runner_mapped():
    index = (ROOT / "experiments/INDEX.md").read_text()
    keys = re.findall(r"^\| ([^|]+) \|", index, re.MULTILINE)
    assert keys.count("d46_oracle_utility_reliability") == 1
    assert "run_d46_oracle_utility_reliability.py" in (ROOT / "scripts/README.md").read_text()


def test_d46_readme_contract():
    readme = (ROOT / "experiments/d46_oracle_utility_reliability/README.md").read_text()
    for section in "ABCDEFGHIJKLM":
        assert f"## {section}." in readme
