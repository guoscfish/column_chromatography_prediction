"""Reconstruct all fitted posteriors and add absolute template diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from run_phase3 import HERE, PLAN, load_state, dump, file_hash, summarize
from kernel_fusion import AdditiveKernelGP, acquisition_diagnostics


def main():
    started = time.perf_counter()
    checks = []
    for seed in PLAN["seeds"]:
        for budget in PLAN["budgets"]:
            state, data, x, xb, audit = load_state(seed, budget)
            n, vs = budget, budget+256
            for path in sorted((HERE/"phase3/results"/f"seed_{seed}_L{budget}").glob("*.json")):
                r = json.loads(path.read_text())
                if r.get("status") != "complete":
                    continue
                j = int(r["metrics"]["endpoint"][-1])-1
                gp = AdditiveKernelGP.restore_from_audit(x[:n], data["train_y"][:, j], r["fit"])
                mu, var = gp.predict(x[vs:])
                predictions = r["predictions"]
                mean_difference = float(np.max(np.abs(mu-[q["mean_ml"] for q in predictions])))
                var_difference = float(np.max(np.abs(var-[q["latent_variance_ml2"] for q in predictions])))
                if mean_difference > 1e-10 or var_difference > 1e-10:
                    raise ValueError("posterior reconstruction mismatch")
                base = r["acquisition"]
                other = acquisition_diagnostics(gp, xb[n:vs], batch_size=32, candidate_limit=256, seed=seed)
                a, b = np.array(base["latent_variance"]), np.array(other["latent_variance"])
                template = dict(policy="fit A on L_t; apply frozen transform and GP to B candidate text; no B fitting",
                                variance_rank_spearman=float(spearmanr(a, b).statistic),
                                mean_absolute_change_ml=float(np.mean(np.abs(np.array(base["latent_mean"])-other["latent_mean"]))),
                                latent_variance_mean_A=float(a.mean()), latent_variance_mean_B=float(b.mean()),
                                mean_absolute_variance_change_ml2=float(np.mean(np.abs(a-b))),
                                latent_mean=other["latent_mean"], latent_variance=other["latent_variance"], methods=other["methods"])
                for name in base["methods"]:
                    aa = set(base["methods"][name]["selected_candidate_positions"])
                    bb = set(other["methods"][name]["selected_candidate_positions"])
                    template[name+"_top32_overlap"] = len(aa & bb)/32
                previous_hash = file_hash(path)
                r["acquisition"]["template_B"] = template
                dump(path, r)
                checks.append(dict(seed=seed, budget=budget, endpoint=r["metrics"]["endpoint"], method=r["metrics"]["method"],
                                   max_mean_difference_ml=mean_difference, max_variance_difference_ml2=var_difference,
                                   result_previous_sha256=previous_hash, result_enriched_sha256=file_hash(path)))
    dump(HERE/"phase3/reproduction.json", dict(passed=True, posteriors_reconstructed=len(checks),
         no_optimizer_calls=True, no_new_labels=True, no_model_selection=True, checks=checks,
         seconds=time.perf_counter()-started))
    summarize()


if __name__ == "__main__":
    main()
