"""Stage-two isolation adapter for the byte-frozen original GNN runner.

No original training function, runner source, module global or stage-one artifact
is changed. A private runner instance receives a support facade whose verifier
checks both seals and whose writes are confined to the new stage-two directory.
The actual nested-calibration acquisition callback is supplied by stage two.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import study_support as stage1_support

STAGE1=Path(__file__).resolve().parent
FAMILIES={'gp_rbf','gp_matern32','blr_independent','blr_coregionalized'}


def _inside(path,root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def validate_stage2_path(stage2):
    stage2=Path(stage2).resolve()
    if stage2==STAGE1.resolve() or not _inside(stage2,STAGE1):
        raise PermissionError('stage two must use a distinct child directory of the EPIG study')
    return stage2


def _verify_both(stage2,verify_stage2_freeze):
    """The caller verifier checks stage-two files/data/packages; parent is exact."""
    stage1_support.verify_freeze()
    returned=verify_stage2_freeze()
    record=stage1_support.read(stage2/'freeze.json')
    if returned is not None and returned!=record:
        raise RuntimeError('stage-two verifier returned a different freeze record')
    if record.get('stage1_freeze_sha256')!=stage1_support.digest(STAGE1/'freeze.json'):
        raise RuntimeError('stage-two freeze does not bind the original stage-one freeze')
    for key in ('git_commit','files','data_hashes','packages'):
        if key not in record:
            raise RuntimeError(f'stage-two freeze missing {key}')
    if not record['files']:
        raise RuntimeError('stage-two freeze has no protected implementation files')
    return record


def load_isolated_runner(stage2,verify_stage2_freeze):
    """Load identical runner bytes with a private global namespace and facade."""
    stage2=validate_stage2_path(stage2)
    _verify_both(stage2,verify_stage2_freeze)
    path=STAGE1/'epig_formal_runner.py'
    parent=stage1_support.read(STAGE1/'freeze.json')
    relative=str(path.relative_to(stage1_support.ROOT))
    if parent['files'].get(relative)!=stage1_support.digest(path):
        raise RuntimeError('original formal runner bytes differ from stage-one seal')
    spec=importlib.util.spec_from_file_location('epig_stage2_private_formal_runner',path)
    private=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private)
    facade=SimpleNamespace(**{name:getattr(stage1_support,name) for name in dir(stage1_support) if not name.startswith('__')})
    def verify():
        return _verify_both(stage2,verify_stage2_freeze)
    def confined_writer(writer):
        def write(path,value):
            if not _inside(path,stage2):
                raise PermissionError(f'stage-two runner write escapes its isolated directory: {path}')
            return writer(path,value)
        return write
    def frozen_feature_matrix():
        # The legacy helper attempts write_once on the already sealed stage-one
        # feature receipt. Requiring it to exist and remain byte-identical makes
        # the call read-only for stage one; only raw X and molecular graphs load.
        path=STAGE1/'features.json'
        before=stage1_support.digest(path)
        frame,x=stage1_support.feature_matrix()
        if stage1_support.digest(path)!=before:
            raise RuntimeError('legacy feature loader altered stage-one receipt')
        facade.write_once(stage2/'features.json',stage1_support.read(path))
        return frame,x
    facade.HERE=stage2
    facade.verify_freeze=verify
    facade.dump=confined_writer(stage1_support.dump)
    facade.write_once=confined_writer(stage1_support.write_once)
    facade.feature_matrix=frozen_feature_matrix
    private.support=facade
    return private


def run_stage2_formal(acquire_callback,stage2_dir,verify_stage2_freeze):
    """Execute only after the unchanged gate passes, preserving 210/40h limits.

    The stage-two decision retains plain family names and posterior_* arm names.
    The callback must separately verify its decision/final calibration receipt
    and use the fixed, registered L333 D on every subsequent authorized-L fit.
    Qualification failure is returned before label contexts or GNN training.
    """
    stage2=validate_stage2_path(stage2_dir)
    runner=load_isolated_runner(stage2,verify_stage2_freeze)
    # This checks all five independent seed decisions, stage-two seal binding,
    # original seeds/arms/row schedule and all original compute limits.
    runner.qualification_gate(stage2)
    return runner.run_formal(acquire_callback,study=stage2)
