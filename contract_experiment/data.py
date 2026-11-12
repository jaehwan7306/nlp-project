"""Official-data validation, duplicate review, and lossless source units."""

import itertools
import re
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from . import LABELS
from .storage import digest, file_hash, now, read_csv, read_json, write_csv, write_json

DATA_URL = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
SPLITS = ("train", "dev", "test")
EXPECTED_COUNTS = {"train": 423, "dev": 61, "test": 123}
NORMALIZATION = "Unicode NFKC, casefold, collapse all whitespace; duplicate detection only"
SIMILARITY = "Jaccard similarity of sets of consecutive 5-word shingles on normalized text"


def fetch_data(destination):
    dest = Path(destination)
    if dest.exists() and any(dest.iterdir()):
        manifest = read_json(dest / "download.json")
        if manifest["url"] != DATA_URL or any(file_hash(dest / name) != sha for name, sha in manifest["files"].items()):
            raise ValueError("Existing data does not match its download manifest. Choose a new destination.")
        print("Verified existing official files; no files overwritten.", flush=True)
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official archive: {DATA_URL}", flush=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "contract-nli.zip"
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "ContractNLI-main-experiment"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        with zipfile.ZipFile(archive) as zipped:
            members = {}
            for split in SPLITS:
                matches = [item for item in zipped.infolist() if Path(item.filename).name == f"{split}.json"]
                if len(matches) != 1 or matches[0].file_size > 150_000_000:
                    raise ValueError(f"Unexpected archive member for {split}.")
                members[split] = matches[0]
            # No extractall: copy only the three expected JSON files.
            for split, member in members.items():
                (dest / f"{split}.json").write_bytes(zipped.read(member))
        write_json(dest / "download.json", {
            "url": DATA_URL, "acquired_utc": now(), "archive_sha256": file_hash(archive),
            "files": {f"{split}.json": file_hash(dest / f"{split}.json") for split in SPLITS},
        })
    print(f"Official files saved to {dest.resolve()}", flush=True)


def build_units(text, spans):
    """Keep complete official spans; fill substantive gaps without editing source text."""
    units, cursor = [], 0
    for span_index, (start, end) in enumerate(spans):
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or start < cursor:
            raise ValueError(f"Invalid, unsorted, or overlapping span {span_index}: {(start, end)}")
        if text[cursor:start].strip():
            units.append({"start": cursor, "end": start, "kind": "gap", "official_span_index": None})
        units.append({"start": start, "end": end, "kind": "official", "official_span_index": span_index})
        cursor = end
    if text[cursor:].strip():
        units.append({"start": cursor, "end": len(text), "kind": "gap", "official_span_index": None})
    for number, unit in enumerate(units, 1):
        unit.update(id=number, text=text[unit["start"]:unit["end"]])
    return units


def load_official(folder):
    documents, hypotheses, warnings = [], None, []
    seen_ids = set()
    hashes = {}
    for split in SPLITS:
        path = Path(folder) / f"{split}.json"
        payload = read_json(path)
        hashes[split] = file_hash(path)
        labels = payload["labels"]
        if len(labels) != 17:
            raise ValueError(f"{split}: expected 17 hypotheses, found {len(labels)}")
        current = {hid: item["hypothesis"] for hid, item in labels.items()}
        if any(not isinstance(text, str) or not text.strip() for text in current.values()):
            raise ValueError(f"{split}: missing hypothesis text")
        if hypotheses is not None and current != hypotheses:
            raise ValueError("Hypotheses differ between splits.")
        hypotheses = current
        if len(payload["documents"]) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected official {split} count: {len(payload['documents'])}")
        for raw in payload["documents"]:
            doc_id, text, spans = raw["id"], raw["text"], raw["spans"]
            if type(doc_id) is not int or doc_id in seen_ids:
                raise ValueError(f"Duplicate/noninteger document ID: {doc_id}")
            seen_ids.add(doc_id)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty/corrupt document: {split}:{doc_id}")
            if not isinstance(spans, list) or any(not isinstance(s, list) or len(s) != 2 for s in spans):
                raise ValueError(f"Invalid spans: {split}:{doc_id}")
            units = build_units(text, spans)
            sets = raw["annotation_sets"]
            if len(sets) != 1:
                raise ValueError(f"Expected one annotation set: {split}:{doc_id}")
            annotations = sets[0]["annotations"]
            if set(annotations) != set(hypotheses):
                raise ValueError(f"Missing/unexpected annotations: {split}:{doc_id}")
            for hid, annotation in annotations.items():
                evidence = annotation["spans"]
                if annotation["choice"] not in LABELS or not isinstance(evidence, list):
                    raise ValueError(f"Invalid annotation: {split}:{doc_id}/{hid}")
                if any(type(sid) is not int or not 0 <= sid < len(spans) for sid in evidence):
                    raise ValueError(f"Invalid evidence reference: {split}:{doc_id}/{hid}")
                if annotation["choice"] == "NotMentioned" and evidence:
                    raise ValueError(f"NotMentioned has evidence: {split}:{doc_id}/{hid}")
            gap_chars = sum(len(unit["text"].strip()) for unit in units if unit["kind"] == "gap")
            if gap_chars or any(len(unit["text"].strip()) <= 2 for unit in units):
                warnings.append({"document": f"{split}:{doc_id}", "gap_characters": gap_chars,
                                 "short_units": sum(len(u["text"].strip()) <= 2 for u in units)})
            documents.append({"id": doc_id, "split": split, "key": f"{split}:{doc_id}",
                              "text": text, "spans": spans, "units": units, "annotations": annotations})
    return {"hypotheses": hypotheses, "documents": documents, "source_hashes": hashes, "unit_warnings": warnings}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    words = text.split()
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}


def label_conflicts(left, right):
    return [hid for hid in left["annotations"]
            if left["annotations"][hid]["choice"] != right["annotations"][hid]["choice"]]


def evidence_texts(doc):
    return {hid: sorted(doc["text"][doc["spans"][sid][0]:doc["spans"][sid][1]]
                        for sid in annotation["spans"])
            for hid, annotation in doc["annotations"].items()}


def audit_data(folder, output, threshold=0.95):
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1].")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Audit directory is not empty. Use a new directory; preserve human reviews.")
    dataset = load_official(folder)
    docs = dataset["documents"]
    normalized = [normalize(doc["text"]) for doc in docs]
    features = [shingles(text) for text in normalized]
    pairs = []
    for i, j in itertools.combinations(range(len(docs)), 2):
        left, right = docs[i], docs[j]
        if left["split"] == right["split"] == "train":
            continue
        if normalized[i] == normalized[j]:
            similarity = 1.0
        else:
            a, b = features[i], features[j]
            if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            intersection = len(a & b)
            similarity = intersection / (len(a) + len(b) - intersection)
        if similarity < threshold:
            continue
        kind = "raw_exact" if left["text"] == right["text"] else (
            "normalized_exact" if normalized[i] == normalized[j] else "similar")
        pair = {"pair_id": "|".join(sorted((left["key"], right["key"]))),
                "left": left["key"], "right": right["key"], "kind": kind,
                "similarity": similarity, "label_conflicts": label_conflicts(left, right),
                "evidence_text_equal": evidence_texts(left) == evidence_texts(right),
                "review_required": "train" not in (left["split"], right["split"])}
        pairs.append(pair)
    out.mkdir(parents=True, exist_ok=True)
    audit = {"created_utc": now(), "data_directory": str(Path(folder).resolve()),
             "source_hashes": dataset["source_hashes"], "threshold": threshold,
             "normalization": NORMALIZATION, "similarity_method": SIMILARITY, "pairs": pairs}
    audit["audit_digest"] = digest(audit)
    write_json(out / "audit.json", audit)
    write_csv(out / "review.csv", [
        {"pair_id": p["pair_id"], "kind": p["kind"], "similarity": f"{p['similarity']:.6f}",
         "label_conflicts": ";".join(p["label_conflicts"]), "review_required": p["review_required"],
         "decision": "", "notes": ""} for p in pairs
    ], ["pair_id", "kind", "similarity", "label_conflicts", "review_required", "decision", "notes"])
    relevant = {key for p in pairs for key in (p["left"], p["right"])}
    write_json(out / "candidate_documents.json", {
        "hypotheses": dataset["hypotheses"], "documents": [d for d in docs if d["key"] in relevant]})
    write_json(out / "unit_checks.json", {"rule": "complete official spans plus substantive uncovered gaps",
                                          "warnings": dataset["unit_warnings"]})
    print(f"Audit: {len(pairs)} candidates; {sum(p['review_required'] for p in pairs)} require human review. No exclusions yet.")


def exclusion_decisions(documents, pairs, reviews):
    """Human decisions form groups; label conflicts exclude the entire confirmed group."""
    docs = {doc["key"]: doc for doc in documents}
    if len(reviews) != len({row["pair_id"] for row in reviews}):
        raise ValueError("Duplicate pair IDs in review.csv")
    review = {row["pair_id"]: row for row in reviews}
    if set(review) != {pair["pair_id"] for pair in pairs}:
        raise ValueError("Review rows do not match the audit candidates.")
    parents = {key: key for key, doc in docs.items() if doc["split"] != "train"}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for pair in pairs:
        if not pair["review_required"]:
            continue
        row = review[pair["pair_id"]]
        decision = row["decision"].strip().lower()
        if decision not in {"duplicate", "distinct"} or not row["notes"].strip():
            raise ValueError(f"Review {pair['pair_id']}: set decision=duplicate/distinct and write notes.")
        if decision == "duplicate":
            parents[root(pair["right"])] = root(pair["left"])
    groups = defaultdict(list)
    for key in parents:
        groups[root(key)].append(docs[key])
    exclusions = []
    for members in groups.values():
        if len(members) < 2:
            continue
        related = ";".join(sorted(doc["key"] for doc in members))
        conflict = any(label_conflicts(members[0], doc) for doc in members[1:])
        tests = [doc for doc in members if doc["split"] == "test"]
        keep = None if conflict else min(tests or members, key=lambda doc: doc["id"])["key"]
        reason = "confirmed_same_content_label_conflict" if conflict else "confirmed_duplicate_keep_test_then_smallest_id"
        for doc in members:
            if doc["key"] != keep:
                exclusions.append({"document": doc["key"], "related_ids": related,
                                   "kept_document": keep or "", "reason": reason})
    return sorted(exclusions, key=lambda row: row["document"])


def prepare_data(audit_path, review_path, output):
    audit = read_json(audit_path)
    claimed = audit.pop("audit_digest")
    if digest(audit) != claimed:
        raise ValueError("Audit file changed. Generate a fresh audit instead of editing audit.json.")
    dataset = load_official(audit["data_directory"])
    if dataset["source_hashes"] != audit["source_hashes"]:
        raise ValueError("Official files changed since the audit.")
    reviews = read_csv(review_path)
    exclusions = exclusion_decisions(dataset["documents"], audit["pairs"], reviews)
    excluded = {row["document"] for row in exclusions}
    documents = [d for d in dataset["documents"] if d["split"] != "train" and d["key"] not in excluded]
    if not all(any(doc["split"] == split for doc in documents) for split in ("dev", "test")):
        raise ValueError("Preprocessing left an empty Dev or Test split.")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prepared directory already contains files. Choose a new output directory.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset.json", {"hypotheses": dataset["hypotheses"], "documents": documents})
    write_json(out / "review_snapshot.json", {"audit": audit, "review_rows": reviews})
    write_csv(out / "exclusions.csv", exclusions, ["document", "related_ids", "kept_document", "reason"])
    write_json(out / "manifest.json", {
        "created_utc": now(), "source_hashes": dataset["source_hashes"], "audit_digest": claimed,
        "review_sha256": file_hash(review_path), "exclusions_sha256": file_hash(out / "exclusions.csv"),
        "dataset_sha256": file_hash(out / "dataset.json"), "review_snapshot_sha256": file_hash(out / "review_snapshot.json"),
        "counts": dict(Counter(doc["split"] for doc in documents)), "excluded": exclusions,
        "unit_rule": "complete official spans plus substantive gaps; concatenate selected units in source order with two newlines",
    })
    counts = Counter(doc["split"] for doc in documents)
    print(f"Prepared: Dev={counts['dev']}, Test={counts['test']}; Test main judgments={5 * 17 * counts['test']}")


def load_prepared(folder):
    path = Path(folder)
    manifest = read_json(path / "manifest.json")
    for name, key in (("dataset.json", "dataset_sha256"), ("exclusions.csv", "exclusions_sha256"),
                      ("review_snapshot.json", "review_snapshot_sha256")):
        if file_hash(path / name) != manifest[key]:
            raise ValueError(f"Prepared file changed: {name}")
    return read_json(path / "dataset.json"), manifest
