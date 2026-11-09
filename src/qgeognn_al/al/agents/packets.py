"""Identity-bound provided-plan packets; no prompt/provider compatibility claim."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import re
from ...infra.contracts import budget_value, content_hash, sample_ids
from .plans import CATALOGUE_FIELDS, filtered_catalogue, validate_plan


def make_packet(rows: Sequence[Mapping[str, object]], *, candidate_ids: Sequence[str],
                seed: int, round_index: int, budget: int, protocol_hash: str):
    """Build a label-free packet for supplied candidate plans, not an old LLM API.

    Full historical Free/CW/NoCW/V3 prompts and transports remain frozen source
    profiles. This packet implements only their common authorization principle:
    choices belong to the exact permitted U identities and bind one seed/round.
    """
    ids = sample_ids(candidate_ids, name='packet candidates')
    budget_value(budget, len(ids))
    if type(seed) is not int or seed < 0 or type(round_index) is not int or round_index < 0:
        raise ValueError('packet requires explicit nonnegative seed/round')
    if not isinstance(protocol_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', protocol_hash):
        raise ValueError('packet requires a pinned protocol SHA-256')
    if any(not isinstance(row, Mapping) or set(row) - CATALOGUE_FIELDS for row in rows):
        raise ValueError('packet candidates may contain only approved label-free metadata')
    packet = {'schema_version': 1, 'protocol': 'provided_authorized_plan_v1', 'seed': seed,
              'round': round_index, 'budget': budget, 'protocol_hash': protocol_hash,
              'candidate_ids': list(ids), 'catalogue': filtered_catalogue(rows, candidate_ids=ids),
              'provider_called': False, 'candidate_labels_available': False}
    return dict(packet, packet_sha256=content_hash(packet))


def validate_bound_plan(response: Mapping[str, object], packet: Mapping[str, object]):
    if not isinstance(response, Mapping) or set(response) != {'packet_sha256', 'selected_ids'}:
        raise ValueError('provided plan must contain only packet binding and selected IDs')
    body = {key: value for key, value in packet.items() if key != 'packet_sha256'}
    if (content_hash(body) != packet.get('packet_sha256')
            or response['packet_sha256'] != packet.get('packet_sha256')):
        raise ValueError('provided plan or packet binding mismatch')
    if packet.get('protocol') != 'provided_authorized_plan_v1' or packet.get('candidate_labels_available') is not False:
        raise ValueError('unregistered or label-bearing candidate packet')
    return validate_plan({'selected_ids': response['selected_ids']},
                         candidate_ids=packet['candidate_ids'], budget=packet['budget'])
