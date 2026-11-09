"""Recreate permitted snapshots from main ancestry and already authorized resources.

Writes only a fresh directory outside the repository. Source rows are gated by
seed/role/identity before any target conversion. No test/U target is exported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
FEATURES=['sample_id','canonical_smiles','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul','PE/EA']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def permitted_rows(source,roles,positions):
    """Never convert any target until its identity has passed the role gate."""
    allowed={identity for identity,role in roles.items() if role in ['l0','validation']}
    rows=[];seen=set()
    with Path(source).open() as stream:
        for row in csv.DictReader(stream):
            identity=row['sample_id']
            if identity not in allowed:continue
            if identity in seen:raise ValueError('duplicate authorized source identity')
            seen.add(identity)
            item={key:row[key] for key in FEATURES}
            item.update(canonical_index=positions[identity],role=roles[identity],V1_ml=float(row['V1_ml']),V2_ml=float(row['V2_ml']))
            rows.append(item)
    if seen!=allowed:raise ValueError('source does not cover authorized identities')
    return rows


def prepare(repository,destination,provenance=HERE/'INPUT_AND_CODE_HASHES.json'):
    repository,destination=Path(repository).resolve(),Path(destination).resolve()
    if destination==repository or repository in destination.parents or destination.exists():
        raise ValueError('snapshot destination must be fresh and outside repository')
    info=json.loads(Path(provenance).read_text());recovery=info['main_ancestor_recovery'];template=info['inputs']
    destination.mkdir(parents=True);code=destination/'code';data=destination/'data';code.mkdir();data.mkdir()
    def relative_source(name):
        pure=PurePosixPath(name)
        if pure.is_absolute() or '..' in pure.parts:raise ValueError('source must be repository-relative')
        return repository/name
    for name,digest in recovery['files'].items():
        content=subprocess.check_output(['git','--no-optional-locks','-C',str(repository),'show',recovery['main_ancestor_recovery_commit']+':'+name])
        if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('historical source identity mismatch: '+name)
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content);target.chmod(0o444)
    identity={'commit':recovery['main_ancestor_recovery_commit'],'equivalent_measured_code_commit':recovery['measured_code_commit'],'files':recovery['files']}
    (code/'SOURCE_IDENTITY.json').write_text(json.dumps(identity,indent=2)+'\n');(code/'SOURCE_IDENTITY.json').chmod(0o444)
    for path in sorted((p for p in code.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.chmod(0o555)
    code.chmod(0o555)
    def verified_copy(name,digest,target):
        source=relative_source(name)
        if sha(source)!=digest:raise ValueError('resource identity mismatch: '+name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if sha(target)!=digest:raise ValueError('copy changed resource bytes')
        target.chmod(0o444)
        return {'source':name,'sha256':digest,'snapshot':str(target),'copied_without_changes':True}
    source_csv=next(r for r in template['source_files'] if r['source'].endswith('canonical_4g.csv'))
    if sha(relative_source(source_csv['source']))!=source_csv['sha256']:raise ValueError('canonical source hash mismatch')
    cache=next(r for r in template['source_files'] if r.get('snapshot','').endswith('graph_cache_4g.pt'))
    manifest={key:value for key,value in template.items() if key not in ['seeds','source_files','source_code_commit','feature_only_u0_policy']}
    manifest['source_code_commit']=recovery['main_ancestor_recovery_commit'];manifest['source_files']=[source_csv,verified_copy(cache['source'],cache['sha256'],data/'graph_cache_4g.pt')];manifest['seeds']=[]
    for item in template['seeds']:
        partition=relative_source(item['source_partition'])
        if sha(partition)!=item['partition_sha256']:raise ValueError('partition hash mismatch')
        with partition.open() as stream:parts=list(csv.DictReader(stream))
        roles={row['sample_id']:row['role'] for row in parts};positions={row['sample_id']:int(row['canonical_index']) for row in parts}
        rows=permitted_rows(relative_source(source_csv['source']),roles,positions)
        if sum(r['role']=='l0' for r in rows)!=333 or sum(r['role']=='validation' for r in rows)!=416:raise ValueError('expected L333 plus 416 validation rows')
        folder=data/f"seed_{item['seed']}";folder.mkdir();target=folder/'authorized_rows.csv'
        with target.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FEATURES+['canonical_index','role','V1_ml','V2_ml'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
        target.chmod(0o444)
        if sha(target)!=item['authorized_rows_sha256']:raise ValueError('filtered snapshot differs from measured authorized rows')
        checkpoint=verified_copy(item['checkpoint']['source'],item['checkpoint']['sha256'],folder/'checkpoint.pt');manifest['source_files'].append(checkpoint)
        updated={k:v for k,v in item.items() if not k.startswith('label_free_u0')};updated.update(authorized_rows=str(target),checkpoint=checkpoint);manifest['seeds'].append(updated)
    result=data/'MANIFEST.json';result.write_text(json.dumps(manifest,indent=2)+'\n');result.chmod(0o444)
    return {'source_code_commit':manifest['source_code_commit'],'source_snapshot':str(code),'snapshot_manifest':str(result),'U_truth_values_converted':0,'test_truth_values_converted':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repository',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);args=p.parse_args();print(json.dumps(prepare(args.repository,args.destination),indent=2))
