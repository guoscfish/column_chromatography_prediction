"""Reuse stable evidence primitives; require explicit review of stale/direct beliefs."""
from ..active_learning_v3.memory import *

def hypotheses_requiring_review(ledger, round_index, latest_ids=(), previous_choices=()):
    latest = set(latest_ids)
    linked = {c['hypothesis_id'] for c in previous_choices if c['id'] in latest and c.get('hypothesis_id')}
    result = []
    for ident, h in sorted(ledger['hypotheses'].items()):
        direct = ident in linked or bool(latest & set(h['supporting_observed_ids'] + h['contradicting_observed_ids']))
        stale = h['status'] != 'rejected' and round_index - h['last_updated_round'] >= 2
        if direct or stale:
            result.append({'id': ident, 'latest_direct_evidence': direct, 'stale_two_rounds': stale,
                           'allowed_actions': ['retain', 'revise', 'support', 'weaken', 'reject']})
    return result
