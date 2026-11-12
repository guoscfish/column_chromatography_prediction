"""Run checked recorded-response selection replays without historical imports."""
from __future__ import annotations
import copy
import gzip
import hashlib
import json
import re
import stat
from pathlib import Path
from .profiles import get_profile
from .replay_packets import replay_recorded_case


def _frozen_artifact_key(name, record, artifacts):
    """Resolve copied basenames to one original freeze key without fuzzy guesses."""
    explicit=record.get('frozen_artifact_key')
    if explicit is not None:
        if explicit not in artifacts:
            raise ValueError('declared frozen artifact key is absent from the source freeze')
        return explicit
    if name in artifacts:
        return name
    source=Path(record.get('source_path',name)).as_posix()
    aliases={name,Path(source).name}
    matches=[key for key in artifacts if Path(key).name in aliases]
    if len(matches)>1:
        exact=[key for key in matches if source==key or source.endswith('/'+str(key).lstrip('/'))]
        if len(exact)==1:return exact[0]
        raise ValueError('ambiguous frozen artifact basename; explicit source key required')
    return matches[0] if matches else None


def _private_case(case, manifest_root):
    """Hydrate only independently copied read-only resources, never live paths."""
    from ...infra.resources import SnapshotFile
    manifest_root=Path(manifest_root).resolve()
    hydrated=copy.deepcopy(case)
    files={}
    for name, record in case['files'].items():
        allowed=name in {'input.json','catalog.json','contract.json','selection.json','batch_freeze.json'} or re.fullmatch(r'(request|turn|rejected|rejected_turn)_\d+\.json',name)
        if not allowed or Path(name).name!=name or record.get('source_kind')!='private_snapshot':
            raise ValueError('private fixture requires literal independent source files')
        resource=record['resource']
        if set(resource)!={'path','root','sha256'} or Path(resource['root']).resolve()!=manifest_root:
            raise ValueError('private fixture resource must stay under its own manifest root')
        item=SnapshotFile(Path(resource['path']),resource['sha256'],manifest_root)
        raw=item.verify().read_bytes()
        if len(raw)>20_000_000:
            raise ValueError('private source record exceeds explicit size limit')
        actual_sha256=hashlib.sha256(raw).hexdigest()
        if actual_sha256!=resource['sha256']:
            raise ValueError('private source resource changed after SHA verification')
        files[name]={'text':raw.decode(),'sha256':actual_sha256,'source_kind':'private_snapshot'}
    freeze=files['batch_freeze.json'];anchor=case['tracked_freeze_anchor'];raw=freeze['text'].encode()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if hashlib.sha256(raw).hexdigest()!=anchor['sha256'] or blob!=anchor['git_blob']:
        raise ValueError('private fixture no longer matches its tracked freeze anchor')
    artifacts=json.loads(freeze['text']).get('artifact_hashes')
    if not isinstance(artifacts,dict) or any(not isinstance(k,str) or not isinstance(v,str)
            or not re.fullmatch('[0-9a-f]{64}',v) for k,v in artifacts.items()):
        raise ValueError('source freeze requires a valid artifact SHA256 map')
    coverage={'tracked_git_freeze':anchor,'direct_artifact_files':{},'not_directly_anchored_files':{}}
    for name,record in files.items():
        if name=='batch_freeze.json':continue
        key=_frozen_artifact_key(name,case['files'][name],artifacts)
        if key is not None:
            if record['sha256']!=artifacts[key]:
                raise ValueError('private source differs from tracked freeze artifact hash: '+name)
            coverage['direct_artifact_files'][name]={'frozen_artifact_key':key,'sha256':record['sha256']}
        elif name.startswith(('rejected_','rejected_turn_')):
            # Some Scientist freezes omit the correction sidecar itself. Its
            # parsed error/instruction is compared during replay, and the next
            # anchored turn binds the exact correction message/config hash.
            coverage['not_directly_anchored_files'][name]={
                'status':'NOT_DIRECTLY_ANCHORED_TO_FREEZE',
                'boundary':'file bytes are snapshot-SHA checked; parsed correction is verified against source validator and next anchored request hash during replay'}
        else:
            raise ValueError('private source file has no frozen artifact hash anchor: '+name)
    # These constructors were source-audited before snapshotting: no credential,
    # header or raw reasoning file is part of this allowlist.
    forbidden={'api_key','authorization','headers','password','access_token','secret','raw_reasoning'}
    def check(value):
        if isinstance(value,dict):
            if any(str(k).lower() in forbidden for k in value):
                raise ValueError('unexpected secret/header/raw-reasoning fields in source-visible recording')
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    for name,record in files.items():
        if name.startswith('request_'):
            request=json.loads(record['text'])
            if set(request)!={'messages','config'}:
                raise ValueError('private request is not the audited messages/config schema')
            check(request)
        elif name.startswith('turn_'):
            turn=json.loads(record['text'])
            if not {'request_sha256','answer','provenance'}<=set(turn):
                raise ValueError('private turn lacks source-visible answer provenance')
            check(turn)
    if 'prompt' in case:
        record=case['prompt'];raw=record['text'].encode()
        if hashlib.sha256(raw).hexdigest()!=record['sha256'] or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=record['git_blob']:
            raise ValueError('tracked Prompt source identity mismatch')
    hydrated['files']=files
    hydrated['freeze_anchor_coverage']=coverage
    return hydrated


def replay_fixture(path, profile=None):
    """Verify original tracked bytes and replay only the explicitly recorded layer.

    The bundled fixture has 6 screen-v1 rounds across 2 seeds and 2 Scientist V2
    rounds for seed157, including its actual invalid-evidence correction. Missing
    profiles absent from this explicit fixture remain blocked; no inference is made about physical availability elsewhere. Fit/reveal/metrics are outside this selection-only replay.
    """
    path=Path(path)
    if any(p.is_symlink() for p in [path,*path.absolute().parents]):
        raise ValueError('replay manifest symlinks are forbidden')
    manifest_before=path.stat()
    if not stat.S_ISREG(manifest_before.st_mode):
        raise ValueError('replay manifest must be a regular file')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as stream:
        raw=stream.read(50_000_001)
    if len(raw)>50_000_000:
        raise ValueError('replay fixture exceeds the explicit size limit')
    fixture=json.loads(raw)
    if fixture.get('schema_version') not in (1,2) or not isinstance(fixture.get('cases'),list):
        raise ValueError('unsupported recorded-response fixture schema')
    private=fixture['schema_version']==2
    if private:
        manifest_after=path.stat()
        identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_nlink,value.st_mode)
        if (manifest_after.st_nlink!=1 or manifest_after.st_mode & 0o222
                or identity(manifest_before)!=identity(manifest_after)):
            raise ValueError('private replay manifest must be independent read-only and unchanged')
    if private and fixture.get('source_kind')!='private_snapshot':
        raise ValueError('schema2 replay must declare private_snapshot origin')
    requested=get_profile(profile) if profile is not None else None
    receipts=[]
    for original in fixture['cases']:
        p=get_profile(original['profile'])
        if requested is not None and p.name!=requested.name:
            if not (requested.name=='free16_cwgradient_lcmd16' and p.name=='cw16_llm16_scientist_v2'):
                continue
        case=_private_case(original,path.parent) if private else copy.deepcopy(original)
        records=list(case['files'].values())+([case['prompt']] if 'prompt' in case else [])
        for record in records:
            if private:continue
            source=record['text'].encode()
            blob=hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest()
            if record['git_blob']!=blob:
                raise ValueError('recorded Git blob identity mismatch')
        if requested is not None:
            case['profile']=requested.name
        result=replay_recorded_case(case)
        result['source_profile']=original['profile']
        if private:result['freeze_anchor_coverage']=case['freeze_anchor_coverage']
        receipts.append(result)
    return {'status':'EXACT_REPLAY_PASS' if receipts else 'BLOCKED_MISSING_FIXTURE',
            'scope':'recorded selection/packet/Prompt/correction/freeze only; not full real AL trajectory equivalence',
            'source_kind':'private_snapshot' if private else 'tracked_git',
            'source_commit':fixture['source_commit'],'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'profiles':sorted({r['profile'] for r in receipts}), 'cases_replayed':len(receipts),
            'results':receipts,'fit_reveal_metrics':'NOT_VERIFIED_IN_SELECTION_REPLAY',
            'online_science':'ONLINE_SCIENCE_NOT_MEASURED','provider_calls':0,'new_real_labels':0,'new_fits':0}
