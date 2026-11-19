"""Read-only graph inference and shared physical-unit response descriptions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/full_volume_audit_mpl')
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT))
from src.qgeognn_al.data import build_model_data,qg

FIELDS=['PE/EA','Density g/ml','V/ul','loading solvent','Volume of loading solvent/ul']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def metrics(actual,predicted,threshold=.5):
    actual=np.asarray(actual,float);predicted=np.asarray(predicted,float)
    meaningful=np.abs(actual)>threshold;flat=np.abs(predicted)<.1;sign=np.sign(actual)==np.sign(predicted)
    return dict(n_pairs=len(actual),n_meaningful=int(meaningful.sum()),observed_abs_mean=float(np.abs(actual).mean()),
        predicted_abs_mean=float(np.abs(predicted).mean()),difference_mae=float(np.abs(predicted-actual).mean()),
        amplitude_error_mean=float(np.abs(np.abs(predicted)-np.abs(actual)).mean()),
        direction_accuracy=float(sign[meaningful].mean()) if meaningful.any() else None,
        correct_nonflat_fraction=float((sign&~flat)[meaningful].mean()) if meaningful.any() else None,
        wrong_nonflat_fraction=float(((actual*predicted<0)&~flat)[meaningful].mean()) if meaningful.any() else None,
        flat_fraction=float(flat[meaningful].mean()) if meaningful.any() else None,actual_threshold_ml=threshold,predicted_flat_threshold_ml=.1)

def infer(model,frame,cache,scaler,batch_size=128):
    """Rebuild complete physical-condition graphs with explicit zero target sentinels."""
    qg.device=torch.device('cpu');torch.set_num_threads(2);model.eval()
    frame=frame.reset_index(drop=True).copy();frame['V1_ml']=0.;frame['V2_ml']=0.
    atom,angle=build_model_data(frame,cache,pd.DataFrame(),scaler)
    for a,sid in zip(atom,frame.sample_id):a.sample_id=sid
    assert all(torch.count_nonzero(a.y)==0 for a in atom)
    result=[];ids=[]
    with torch.no_grad():
        for start in range(0,len(atom),batch_size):
            a=Batch.from_data_list(atom[start:start+batch_size]);b=Batch.from_data_list(angle[start:start+batch_size])
            expected=frame.sample_id.iloc[start:start+batch_size].tolist();assert a.sample_id==expected
            result.append(model(a,b).numpy());ids.extend(a.sample_id)
    pred=np.vstack(result);assert ids==frame.sample_id.tolist() and np.isfinite(pred).all()
    out=pd.DataFrame(pred,columns=['V1_q10','V1_q50','V1_q90','V2_q10','V2_q50','V2_q90']);out.insert(0,'sample_id',ids)
    return out

def summary(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        for threshold in [.1,.5,1.]:
            rows.append(dict(zip(keys,key),molecules=g.molecule.nunique(),series=g.series_key.nunique(),**metrics(g.delta_true,g.delta_pred,threshold)))
    return pd.DataFrame(rows)
