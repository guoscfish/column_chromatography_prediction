import sys,time,json
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P));import resume_routed_transport_20261016 as routed
m=routed.m
while not (P/'result.json').exists():
 if all((P/f'{seed}_{method}_result.json').exists() for seed in (157,6101) for method in m.METHODS):
  routed.guard()
  for seed in (157,6101):
   for method in m.METHODS:
    result=m.read(P/f'{seed}_{method}_result.json')
    assert result['final_labels']==1549
    assert len(result['selected_ids'])==len(set(result['selected_ids']))==1216
    assert len(result['additional_ids'])==len(set(result['additional_ids']))==128
    for round_index in range(35,39):
     f=m.read(P/f'runtime/seed_{seed}/{method}/round_{round_index:02d}/prediction_freeze.json')
     for k in ('checkpoint','prediction'):assert m.sha(f[k+'_path'])==f[k+'_sha256']
  m.once(P/'result.json',{'status':'ALL_EIGHT_COMPLETE1549','test_evaluation_performed':False,'completed_time':time.time(),'completion_observer':'verified all eight real target results and32 frozen fits; supports safely resumed workers'})
  print('ALL_EIGHT_COMPLETE1549',flush=True);break
 time.sleep(10)
