"""Row science inputs and fit transaction contracts; no real training is run."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.qgeognn_al.predictor import row, row_fit
from src.qgeognn_al.predictor.conditions import ConditionNormalization
from src.qgeognn_al.infra.identity import ids_hash, ordered_ids_hash
from src.qgeognn_al.infra.resources import SnapshotFile

ROOT = Path(__file__).resolve().parents[1]
# Captured from immutable main b1a6ed7a small_batch_benchmark/splits/row_seed_*.csv.
# Values describe ID roles only; no source targets were consumed.
GOLDEN = {'157': {'l0': {'rows': 333,
                'ids_hash': '1171988daf4e91eb3d866b44b1ba07f04a36e6f88eb953eda831b9c64c66854d',
                'ordered_ids_hash': '2e48068e9d55fee6bbcb0a3d4e8fe89c821cc925732f95e06205b7cc66d1d9e7'},
         'u0': {'rows': 2997,
                'ids_hash': '2d5225ceee78487280509f7b1352724a9a87b095ebccf0041d9fe2b5b7e5bf47',
                'ordered_ids_hash': '9875d5046e97390458db6971fc55635092c579e1e753369d66aeb919a92316da'},
         'validation': {'rows': 416,
                        'ids_hash': '1c7360ea30a5bb2bbc04c1160ce749a6592ce89a5314acb6d1aa7bd7ce5d6ead',
                        'ordered_ids_hash': 'f5240cbae93a53113742d693303f994315d73a80355e4fde2987822a3a337865'},
         'test': {'rows': 417,
                  'ids_hash': '10a2ef48671c007be63ae19d9203e2af7c5f13b954f505102564aaeabe0ecc19',
                  'ordered_ids_hash': 'e484cdea2e2e31c0632479df8276206a62154802ee3451b484057c6a0fb7fc90'}},
 '887': {'l0': {'rows': 333,
                'ids_hash': '7c4a495fbada80f6a2894d232c8e9da2d0fbdb72e36460ec8fad3673097b8eaa',
                'ordered_ids_hash': '8c0ef236aa556a8de1b6bee92eef46b9108433d701288b6eb78d7ca61e3d9958'},
         'u0': {'rows': 2997,
                'ids_hash': 'd83c4dca60772a94228fffd1612bd1ff826c9a56e2b25300501caf6e60820fb1',
                'ordered_ids_hash': '4768b8a2f82ef83f4e80acc827a179cdfbe9b34f77232fb6da99435ff5ae4172'},
         'validation': {'rows': 416,
                        'ids_hash': 'c9d69291ee84c64472ce3af1a9d4479ba63acacae4993a663d261576a29306c3',
                        'ordered_ids_hash': 'bf9ea6b42d03a998467d38d8a5101ac3578e6943b1ef9d83a37d579cb17097ab'},
         'test': {'rows': 417,
                  'ids_hash': '990930f043cb4b5a7327059f853719d8ec62ce21eeaf1756acbc315e6c0a33de',
                  'ordered_ids_hash': '2ed9201d0d915031e5d3aef05c17e14c02f997f3d174f15a057248ea35e6ec4a'}},
 '2357': {'l0': {'rows': 333,
                 'ids_hash': 'd779b85de80af8c7c8ad001710859df64ec813a7b86ffc269dce54b110791e7f',
                 'ordered_ids_hash': '146703e076968d0f0ec18c0a663d517096d347fad8dc0731f809556b62ba1946'},
          'u0': {'rows': 2997,
                 'ids_hash': '3f6b3a1e57ba608e9eae9726b16dc79ceccb489a255ac6cda2df786037a105b9',
                 'ordered_ids_hash': '428495885c2d9b4097c46aaa57c19961436ce0b1057c3f6568244a3cccac4637'},
          'validation': {'rows': 416,
                         'ids_hash': 'feec528b1659a310aaa38f4d0a01839349e00247504b3b62a3ae3afa021ecf79',
                         'ordered_ids_hash': 'f2bd057fee1471351d4d2c3edda2bc3a010934c211991df050f06b0b39ece52c'},
          'test': {'rows': 417,
                   'ids_hash': '933364fc5a6cba8a09c8cc1e2f97f23b1e9ab62f52b99bb5fb5695c734fe8e20',
                   'ordered_ids_hash': 'dcc438e07a5ac8c4458875809a7f885dc5cba6d98eb025d1341d8cb1135eecd9'}},
 '6101': {'l0': {'rows': 333,
                 'ids_hash': '38793e6f1eda734757c562de4973830b27cf13b2c0e97fbf110c5a1e61fb860e',
                 'ordered_ids_hash': '6dbdf532541c71db5f04077f28c2045000c524a2e6dd37592dd497e524a4ed4c'},
          'u0': {'rows': 2997,
                 'ids_hash': '2b9590b3a68506f0c2cab57988232d3bc220911d12e680709ae02259f81aa149',
                 'ordered_ids_hash': 'b3d1575e14176c0b9548149527bc4e3b83d28073f99afdaaee72b94c96a456a7'},
          'validation': {'rows': 416,
                         'ids_hash': '247750c1d939b88b0869fc235e77b9db721b357703d85933996189915abaeb8d',
                         'ordered_ids_hash': 'b9fdcf64f67f05981d942abefdf58c984f4a138e11b5a7f07d4106de2a4ad216'},
          'test': {'rows': 417,
                   'ids_hash': 'ecb7ff19582c1a90bb1182476e3e3db6c1655216f9521c04acd90ea593b162f5',
                   'ordered_ids_hash': 'ecbaf32ae1b756641fd429540bd179076668065cf397506760a2c389800863d8'}},
 '12203': {'l0': {'rows': 333,
                  'ids_hash': '71385392dba35d74a79610ae88dddc15c44ad740e0d34c59466888febddf24cb',
                  'ordered_ids_hash': '1e0b0c06296c4e657abebe8fff0a2976c69763830948c6a504a14cb487b05a57'},
           'u0': {'rows': 2997,
                  'ids_hash': 'e29127d3b54cd0668358bc5bf74622fa3967215ce03cda1318bbf8ccbd1e7c78',
                  'ordered_ids_hash': '2db5c3181b5e3b6fc14463eaf3e892cf1ad43988fd1ee8e4e6830a8504a41803'},
           'validation': {'rows': 416,
                          'ids_hash': 'a17378c88592bc6117515c9305a6e9536da627388ce6c275eb143ff3a6d3a491',
                          'ordered_ids_hash': '6db22adbc5dbb2da72b0592ca81f5f11decfdc12d6277d9a1d77e75e5e87f407'},
           'test': {'rows': 417,
                    'ids_hash': '6ea79f93fab3ddf63157b5f71569316647c79a10673e6d42c00aaddbf08ac991',
                    'ordered_ids_hash': '09e01ca3ea99de3af8a28422e5a7254c1e34a8dad4a7bea80b2cb39f0eb5c096'}}}


@pytest.mark.parametrize('seed', row.CONFIRMATION_SEEDS)
def test_five_real_canonical_row_partitions_match_frozen_identity_roles(seed):
    frame = pd.read_csv(ROOT/'experiments/e0_4g_baseline/canonical_4g.csv', usecols=list(row.FEATURE_COLUMNS))
    split = row.make_row_protocol(frame, seed)
    assert len(split) == 4163 and split.canonical_index.tolist() == list(range(4163))
    for role, expected in GOLDEN[str(seed)].items():
        identities = split.loc[split.role.eq(role), 'sample_id'].astype(str)
        assert len(identities) == expected['rows']
        assert ids_hash(identities) == expected['ids_hash']
        assert ordered_ids_hash(identities) == expected['ordered_ids_hash']


def synthetic_dataset():
    frame = pd.DataFrame({'sample_id':[f's{i}' for i in range(20)], 'canonical_smiles':['C']*20,
        'Density g/ml':np.arange(20,dtype=float)+1, 'V/ul':np.arange(20,dtype=float)+2,
        'loading solvent':['PE']*20, 'Volume of loading solvent/ul':np.arange(20,dtype=float)+3,
        'PE/EA':['1/1']*20})
    partition = row.make_row_protocol(frame, 157)
    cache={'C':{'descriptor':np.arange(16,dtype=np.float32)}}
    return frame, partition, cache


def construct(frame, partition, cache):
    return row.RowDataset(frame, partition, cache, source_sha256='1'*64,
                          split_sha256='2'*64, graph_cache_sha256='3'*64)


def test_row_prepare_only_explicit_l0_labels_and_label_free_graphs(monkeypatch,tmp_path):
    frame, partition, cache=synthetic_dataset();dataset=construct(frame,partition,cache)
    outer=np.r_[np.flatnonzero(partition.role.eq('l0')),np.flatnonzero(partition.role.eq('u0'))]
    assert dataset.outer_train_indices == tuple(outer)
    captured=[]
    def builder(features, cache, split, scaler):
        captured.append(features.copy())
        return ([Data(x=torch.zeros((1,1),dtype=torch.long), edge_index=torch.empty((2,0),dtype=torch.long),
                      edge_attr=torch.empty((0,1)), y=torch.zeros((1,2)), canonical_position=torch.tensor(i))
                 for i in range(len(features))],
                [Data(edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,1))) for _ in range(len(features))])
    monkeypatch.setattr(row,'build_model_data',builder)
    prepared=dataset.prepare(np.array([[1.,3.],[5.,11.]]),tmp_path/'scaler.json')
    assert prepared.preprocessing['target_scales']=={'V1':2.,'V2':4.}
    assert prepared.preprocessing['covariate_fit_rows']==16
    assert prepared.l0_ids==dataset.l0_ids and prepared.outer_seed==157
    assert list(prepared.rows)==frame.sample_id.tolist()
    assert all('y' not in item.atom and 'y' not in item.angle for item in prepared.rows.values())
    assert (captured[0][['V1_ml','V2_ml']].to_numpy()==0).all()
    assert prepared.normalization.loading_amount_max==float((frame.iloc[outer]['Density g/ml']*frame.iloc[outer]['V/ul']).max())
    with pytest.raises(ValueError,match='protected'):
        dataset.prepare(np.array([[1.,3.],[5.,11.]]),tmp_path/'protected/scaler.json',forbidden_output_roots=[tmp_path/'protected'])
    assert not (tmp_path/'protected').exists()


def test_row_data_rejects_targets_partition_drift_and_unknown_cache():
    frame, partition, cache=synthetic_dataset()
    with pytest.raises(ValueError,match='label-free'):construct(frame.assign(V1_ml=999),partition,cache)
    with pytest.raises(ValueError,match='partition may contain'):construct(frame,partition.assign(V2_ml=999),cache)
    bad=partition.copy();bad.loc[0,'sample_id']='unrelated'
    with pytest.raises(ValueError,match='identity/order'):construct(frame,bad,cache)
    bad=partition.copy();bad.loc[0,'canonical_index']=1
    with pytest.raises(ValueError,match='canonical indices'):construct(frame,bad,cache)
    with pytest.raises(ValueError,match='graph cache'):construct(frame,partition,{})


def test_row_file_adapter_parses_only_x_and_never_hidden_target_strings(tmp_path):
    frame,partition,cache=synthetic_dataset()
    frame['V1_ml']='HIDDEN_TARGET_MUST_NOT_PARSE';frame['V2_ml']='HIDDEN_TARGET_MUST_NOT_PARSE'
    paths=[tmp_path/'source.csv',tmp_path/'partition.csv',tmp_path/'cache.pt']
    frame.to_csv(paths[0],index=False);partition.to_csv(paths[1],index=False);torch.save(cache,paths[2])
    snapshots=[]
    for p in paths:
        p.chmod(0o444);snapshots.append(SnapshotFile(p,hashlib.sha256(p.read_bytes()).hexdigest(),tmp_path))
    dataset=row.RowDataset.from_files(source=snapshots[0],partition=snapshots[1],graph_cache=snapshots[2])
    assert set(dataset.feature_frame)==set(row.FEATURE_COLUMNS)
    assert 'study' in dataset.partition
    assert dataset.split_sha256==snapshots[1].sha256
    assert dataset.partition_hash!=dataset.split_sha256


@pytest.mark.parametrize('member',[0,1,2])
def test_initialization_and_training_seed_streams_preserve_historical_recipe(member):
    config=row_fit.row_training_config(157,'1'*64,member)
    assert row_fit.initialization_seed(157,member)==157+2_000_003+member*1_000_033
    assert config['seed']==157+3_000_017+member*1_000_033
    assert row_fit.sketch_seed(157)==157+4_000_037
    assert config['learning_rate']==0.001 and config['batch_size']==2048
    assert config['maximum_epochs']==1000 and config['patience']==100
    assert config['checkpoint_selection']=='validation_combined_normalized_rmse'
    assert config['test_during_training'] is False
    config['loss_weights']['V1']=123
    assert row_fit.row_training_config(157,'1'*64,member)['loss_weights']['V1']==1.


def test_actual_model_initialization_is_independent_of_previous_rng_draws():
    norm=ConditionNormalization(0.,10.,0.,10.,'4g','source_train',2,'1'*64,'2'*64)
    first,identity=row_fit.initialize_row_model(norm,157)
    torch.rand(100);np.random.rand(100)
    second,second_identity=row_fit.initialize_row_model(norm,157)
    assert identity==second_identity
    assert all(torch.equal(value,second.state_dict()[key]) for key,value in first.state_dict().items())
    _,other_identity=row_fit.initialize_row_model(norm,157,1)
    assert other_identity!=identity


def fit_arguments(tmp_path):
    atom=[Data(canonical_position=torch.tensor(i)) for i in range(4)]
    return dict(atom_base=atom,angle=[Data() for _ in atom],normalization=ConditionNormalization(0.,1.,0.,1.,'4g','source_train',2,'1'*64,'2'*64),
        preprocessing={'target_scales':{'V1':1.,'V2':2.}},train_indices=[0,1],train_truth=np.array([[1.,2.],[3.,6.]]),
        validation_indices=[2],validation_truth=np.array([[2.,4.]]),prediction_indices=[3,0],prediction_sample_ids=['s3','s0'],
        initialization_seed=row_fit.initialization_seed(157),training_config=row_fit.row_training_config(157,'1'*64),
        contract={'train_sample_ids':['s0','s1'],'validation_sample_ids':['s2']},runtime=tmp_path/'fit')


def test_fresh_real_fit_is_denied_and_partial_fit_fails_closed_before_model(monkeypatch,tmp_path):
    args=fit_arguments(tmp_path)
    def forbidden(*a,**k):raise AssertionError('model must not be constructed before permission/partial-state checks')
    monkeypatch.setattr(row_fit,'build_predictor',forbidden)
    with pytest.raises(PermissionError):row_fit.fit_row_from_initialization(**args)
    assert not args['runtime'].exists()
    args['runtime'].mkdir();started=args['runtime']/'fit_started.json';started.write_text('{"ambiguous":true}')
    with pytest.raises(RuntimeError,match='ambiguous partial'):
        row_fit.fit_row_from_initialization(**args,allow_training=True)
    assert started.read_text()=='{"ambiguous":true}' and len(list(args['runtime'].iterdir()))==1


def test_mock_fit_transaction_reuses_completion_and_refuses_drift(monkeypatch,tmp_path):
    # The original optimizer/training function is never executed in this test.
    args=fit_arguments(tmp_path);events=[]
    monkeypatch.setattr(row_fit,'configure_row_runtime',lambda **kwargs:None)
    monkeypatch.setattr(row_fit,'build_predictor',lambda normalization:torch.nn.Linear(1,1))
    def fit(model, atom, angle, train, validation, preprocessing, config, checkpoint):
        events.append('mock_fit')
        assert atom[0].y.tolist()==[[1.,2.]] and atom[2].y.tolist()==[[2.,4.]]
        assert 'y' not in atom[3] and all('y' not in original for original in args['atom_base'])
        torch.save(model.state_dict(),checkpoint)
        return [{'validation_selection_score':0.25,'epoch':1}],1
    monkeypatch.setattr(row_fit,'train_source',fit)
    def load(path):
        model=torch.nn.Linear(1,1);model.load_state_dict(torch.load(path,weights_only=True));return model
    monkeypatch.setattr(row_fit,'load_predictor_checkpoint',load)
    monkeypatch.setattr(row_fit,'predict_outputs',lambda model,atom,angle,positions,batch:np.arange(12).reshape(2,6))
    first=row_fit.fit_row_from_initialization(**args,allow_training=True)
    (args['runtime']/'best.pt').chmod(0o444)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args['runtime'].iterdir()}
    second=row_fit.fit_row_from_initialization(**args)
    assert first==second and events==['mock_fit']
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args['runtime'].iterdir()}
    assert first['best_epoch']==1 and first['best_validation_combined_normalized_rmse']==0.25
    drift={**args,'training_config':{**args['training_config'],'seed':987}}
    with pytest.raises(RuntimeError,match='incompatible completed'):
        row_fit.fit_row_from_initialization(**drift)
    assert events==['mock_fit']


@pytest.mark.parametrize('name',['best.pt','predictions.csv.gz','fit_audit.json','fit_started.json','history.csv'])
@pytest.mark.parametrize('link_kind',['symlink','hardlink'])
def test_completed_fit_never_borrows_linked_receipt_files(monkeypatch,tmp_path,name,link_kind):
    args=fit_arguments(tmp_path);runtime=args['runtime'];runtime.mkdir()
    # Mark all three completion pieces present. The linked piece must be rejected
    # before parsing any receipt or constructing/loading a predictor.
    for piece in ['best.pt','predictions.csv.gz','fit_audit.json']:
        if piece!=name:(runtime/piece).write_bytes(b'owned placeholder')
    external=tmp_path/'external_owned_file';external.write_bytes(b'outside fit must remain untouched')
    linked=runtime/name
    if link_kind=='symlink':linked.symlink_to(external)
    else:os.link(external,linked)
    inode=external.stat().st_ino;before=external.read_bytes()
    def forbidden(*a,**k):raise AssertionError('must reject link before loading/initializing model')
    monkeypatch.setattr(row_fit,'build_predictor',forbidden)
    with pytest.raises(ValueError,match='symlink|hardlink|independent regular'):
        row_fit.fit_row_from_initialization(**args)
    assert external.stat().st_ino==inode and external.read_bytes()==before


def test_completed_fit_rejects_non_regular_receipt_before_model(monkeypatch,tmp_path):
    args=fit_arguments(tmp_path);runtime=args['runtime'];runtime.mkdir()
    (runtime/'best.pt').mkdir()
    monkeypatch.setattr(row_fit,'build_predictor',lambda *a,**k:pytest.fail('no model before receipt preflight'))
    with pytest.raises(ValueError,match='independent regular'):
        row_fit.fit_row_from_initialization(**args,allow_training=True)
