"""
cleanup.py — Safe deletion of previous-stage artifacts (opt-in).

Rules:
  1. Only delete after the downstream stage manifest is COMPLETE
  2. Verify downstream outputs via checksum + structural validation before delete
  3. Log every deletion to a deletion log for audit trail
  4. Never delete tokenizers, checkpoints, logs, manifests, or packed splits
  5. Dry-run mode available for testing

Intentionally conservative: refuses to delete on any doubt about downstream integrity.
In this framework cleanup is opt-in (--cleanup) because the user's source data
and packed splits are precious.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .manifest import StageManifest, validate_bin_file, file_checksum


# Path fragments that must never be deleted regardless of stage
PROTECTED_PATTERNS = [
    "checkpoints",
    "tokenizer",
    "logs",
    "manifests",
    "_packed.bin",
]


class CleanupError(RuntimeError):
    pass


class StageCleanup:
    def __init__(
        self,
        stage_name: str,
        artifact_dir: str | Path,
        downstream_manifest: StageManifest,
        deletion_log: str | Path = "logs/deletion_log.jsonl",
        seq_len: int = 512,
        dry_run: bool = False,
    ):
        self.stage_name          = stage_name
        self.artifact_dir        = Path(artifact_dir)
        self.downstream_manifest = downstream_manifest
        self.deletion_log        = Path(deletion_log)
        self.seq_len             = seq_len
        self.dry_run             = dry_run

    def run(self) -> dict:
        result = {
            "stage":         self.stage_name,
            "artifact_dir":  str(self.artifact_dir),
            "dry_run":       self.dry_run,
            "deleted_bytes": 0,
            "deleted_files": 0,
            "skipped":       False,
            "error":         "",
            "timestamp":     time.time(),
        }

        if not self.downstream_manifest.is_complete():
            summary = self.downstream_manifest.summary()
            result["skipped"] = True
            result["error"]   = f"downstream manifest not complete: {summary}"
            print(f"[cleanup:{self.stage_name}] SKIP — downstream not complete: {summary}")
            return result

        try:
            self._verify_downstream()
        except CleanupError as e:
            result["skipped"] = True
            result["error"]   = str(e)
            print(f"[cleanup:{self.stage_name}] SKIP — verification failed: {e}")
            return result

        self._assert_not_protected(self.artifact_dir)

        files = list(self._enumerate_deletable())
        total_bytes = sum(f.stat().st_size for f in files if f.exists())

        if not files:
            print(f"[cleanup:{self.stage_name}] Nothing to delete in {self.artifact_dir}")
            return result

        print(
            f"[cleanup:{self.stage_name}] "
            f"{'DRY RUN — ' if self.dry_run else ''}"
            f"Deleting {len(files)} files "
            f"({total_bytes / 1e9:.2f} GB) from {self.artifact_dir}"
        )

        deleted_bytes = 0
        deleted_files = 0
        for f in files:
            if not self.dry_run:
                try:
                    bytes_before = f.stat().st_size
                    f.unlink()
                    deleted_bytes += bytes_before
                    deleted_files += 1
                except OSError as e:
                    print(f"[cleanup:{self.stage_name}] WARNING: could not delete {f}: {e}")
            else:
                deleted_bytes += f.stat().st_size
                deleted_files += 1

        if not self.dry_run:
            self._remove_empty_dirs(self.artifact_dir)

        result["deleted_bytes"] = deleted_bytes
        result["deleted_files"] = deleted_files
        self._log_deletion(result)
        print(
            f"[cleanup:{self.stage_name}] "
            f"{'Would delete' if self.dry_run else 'Deleted'} "
            f"{deleted_files} files, {deleted_bytes / 1e9:.2f} GB"
        )
        return result

    def _verify_downstream(self):
        entries = self.downstream_manifest.verified_entries()
        if not entries:
            raise CleanupError("downstream manifest has no verified entries")

        for entry in entries:
            p = Path(entry.output_path)
            if not p.exists():
                raise CleanupError(
                    f"downstream output missing: {entry.output_path} (shard {entry.shard_id})"
                )
            if entry.checksum:
                actual = file_checksum(p)
                if actual != entry.checksum:
                    raise CleanupError(
                        f"checksum mismatch for {entry.output_path}: "
                        f"expected {entry.checksum[:12]}… got {actual[:12]}…"
                    )
            if p.suffix == ".bin":
                valid, _, err = validate_bin_file(p, self.seq_len)
                if not valid:
                    raise CleanupError(f"invalid .bin file {entry.output_path}: {err}")

    def _enumerate_deletable(self) -> list[Path]:
        if not self.artifact_dir.exists():
            return []
        return [
            f for f in self.artifact_dir.rglob("*")
            if f.is_file() and not self._is_protected(f)
        ]

    def _is_protected(self, path: Path) -> bool:
        path_str = str(path)
        return any(pat in path_str for pat in PROTECTED_PATTERNS)

    def _assert_not_protected(self, path: Path):
        path_str = str(path)
        for pat in PROTECTED_PATTERNS:
            if pat in path_str:
                raise CleanupError(
                    f"Refusing to delete protected path: {path} (matches '{pat}')"
                )

    def _remove_empty_dirs(self, root: Path):
        for dirpath in sorted(root.rglob("*"), reverse=True):
            if dirpath.is_dir():
                try:
                    dirpath.rmdir()
                except OSError:
                    pass

    def _log_deletion(self, record: dict):
        self.deletion_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.deletion_log, "a") as f:
            f.write(json.dumps(record) + "\n")


def safe_delete_stage(
    stage_name: str,
    artifact_dir: str | Path,
    downstream_manifest_path: str | Path,
    deletion_log: str | Path = "logs/deletion_log.jsonl",
    seq_len: int = 512,
    dry_run: bool = False,
) -> dict:
    manifest = StageManifest(downstream_manifest_path)
    cleaner  = StageCleanup(
        stage_name          = stage_name,
        artifact_dir        = artifact_dir,
        downstream_manifest = manifest,
        deletion_log        = deletion_log,
        seq_len             = seq_len,
        dry_run             = dry_run,
    )
    return cleaner.run()
