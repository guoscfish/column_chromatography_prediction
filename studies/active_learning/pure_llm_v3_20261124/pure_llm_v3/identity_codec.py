"""Assign one index per frozen training row and one per canonical structure."""
from dataclasses import dataclass

from .records import digest


@dataclass(frozen=True)
class IdentityCodec:
    rows: dict[str, int]
    molecules: dict[str, str]
    mode: str = "short"

    @classmethod
    def from_pool(cls, structures, mode="short"):
        if mode not in {"short", "legacy"} or not structures or len(structures) > 9999:
            raise ValueError("unsupported identity mode or universe size")
        return cls(
            {row: number for number, row in enumerate(sorted(structures), 1)},
            {s: f"M-{number:04d}" for number, s in enumerate(sorted(set(structures.values())), 1)},
            mode,
        )

    def row(self, sample_id, *, observed):
        number = self.rows[sample_id]
        if self.mode == "legacy":
            return sample_id
        return f"{'O' if observed else 'C'}-{number:04d}"

    @property
    def all_row_ids(self):
        return {self.row(row, observed=observed): row for row in self.rows for observed in (False, True)}

    def mapping_digest(self, observed):
        mapping = {
            self.row(row, observed=row in observed): row for row in self.rows
        }
        return digest({"roles": mapping, "observed": sorted(observed), "molecules": self.molecules})
