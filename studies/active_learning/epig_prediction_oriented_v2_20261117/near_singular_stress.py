"""Report float64 sensitivity against an 80-decimal Gaussian MI oracle.

This is a diagnostic of deliberately ill-conditioned synthetic matrices. It
does not modify acquisition code or the frozen real-posterior comparison.
"""
from pathlib import Path
import json
import sys

import mpmath as mp
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from qgeognn_al.active_learning_v2.epig import gaussian_pair_information

def run():
    mp.mp.dps=80
    rng=np.random.default_rng(390);rows=[]
    for q in (1,2,3):
        for epsilon in (1e-5,1e-8,1e-10,1e-12):
            a,_=np.linalg.qr(rng.normal(size=(q,q)))
            b,_=np.linalg.qr(rng.normal(size=(q,q)))
            cross=a@np.diag(np.linspace(.8,1-epsilon,q))@b.T
            if q==1:cross[:]=1-epsilon
            exact=mp.matrix([[mp.mpf(float(v)) for v in row] for row in cross])
            schur=mp.eye(q)-exact*exact.T
            high=float(-mp.log(mp.det(schur))/2)
            got=float(gaussian_pair_information(np.eye(q),np.eye(q),cross)[0])
            rows.append({"q":q,"singular_gap_design":epsilon,"minimum_float64_schur_eigenvalue":float(np.linalg.eigvalsh(np.eye(q)-cross@cross.T).min()),
                "float64_fast_information":got,"oracle_80_decimal_information":high,"absolute_error_nats":abs(got-high)})
    result={"purpose":"finite precision diagnostic, not a pass/fail calibration or efficacy gate",
        "oracle_decimal_precision":80,"rows":rows,"maximum_absolute_error_nats":max(r["absolute_error_nats"] for r in rows),
        "real_labels_read":0,"new_fits":0}
    (HERE/"near_singular_stress.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__":run()
