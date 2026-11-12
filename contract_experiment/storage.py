"""Small, standard-library helpers for durable experiment files."""

import csv
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path, repair=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows, good_bytes = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            if not repair or index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Invalid journal row: {path}:{index + 1}") from None
            backup = path.with_name(path.name + ".partial-" + datetime.now().strftime("%Y%m%d%H%M%S%f"))
            backup.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(good_bytes)
            return rows
        if not isinstance(row, dict):
            raise ValueError(f"Journal row must be an object: {path}:{index + 1}")
        rows.append(row)
        good_bytes += len(line)
    if repair and data and not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def unique_index(rows, fields):
    index = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate result key: {key}")
        index[key] = row
    return index


@contextmanager
def run_lock(folder):
    """OS-managed lock: automatically released if a process crashes."""
    path = Path(folder) / ".run.lock"
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another command is using this run directory.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
