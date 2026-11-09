"""Config routes into the one workflow and cannot activate scientific fitting."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('run_al',ROOT/'scripts/run_al.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def test_complete_synthetic_config_path(tmp_path):
    config=json.loads((ROOT/'configs/offline_synthetic.json').read_text())
    result=cli.run(config,tmp_path/'run',code_commit='a'*40)
    assert result['status']=='complete' and result['rounds']==2 and result['budget']==6
    assert result['science_training_performed'] is False and result['real_labels_revealed']==0


def test_cli_rejects_scientific_mode_before_backend(tmp_path):
    with pytest.raises(PermissionError):cli.run({'schema_version':2,'execution_mode':'authorized_scientific_run'},tmp_path/'run',code_commit='a'*40)
    assert not (tmp_path/'run').exists()
