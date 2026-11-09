"""Explicit immutable snapshot resources; no fallback into an active checkout."""
from dataclasses import dataclass
from pathlib import Path
import stat
from .artifacts import sha256_file


@dataclass(frozen=True)
class SnapshotFile:
    path: Path
    sha256: str
    root: Path

    def verify(self):
        path, root = Path(self.path).absolute(), Path(self.root).resolve()
        if not path.resolve().is_relative_to(root):
            raise ValueError('snapshot resource must stay within the authorized snapshot root')
        for parent in [path, *path.parents]:
            if parent.is_symlink():
                raise ValueError('snapshot symlinks are forbidden')
            if parent == root:
                break
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise ValueError('snapshot must be an independent read-only regular file')
        if sha256_file(path) != self.sha256:
            raise ValueError('snapshot SHA-256 mismatch')
        return path
