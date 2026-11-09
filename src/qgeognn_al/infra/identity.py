"""Set and sequence identities have distinct explicit scientific semantics."""
from .contracts import content_hash, sample_ids


def ids_hash(values):
    """Historical unordered set identity; stable sorted string normalization."""
    return content_hash(sorted(str(value) for value in values))


def ordered_ids_hash(values):
    return content_hash(list(sample_ids(values, name='ordered identities')))
