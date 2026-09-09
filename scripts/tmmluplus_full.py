"""Write the full TMMLU+ test set as one JSONL in the slice format (quality track, spec §5 W4).

usage: python scripts/tmmluplus_full.py --data-dir <snapshot>/data --out eval/tmmluplus/full.jsonl

Items are ordered by (subject, index) so the file and its SHA-256 are deterministic for a given
dataset snapshot; ``full.json`` next to it records the snapshot, counts per subject and the digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from slo_lab.tmmluplus import load_test_items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", default="ikala/tmmluplus test split (MIT)")
    args = parser.parse_args()
    by_subject = load_test_items(Path(args.data_dir))
    items = [item for subject in sorted(by_subject) for item in by_subject[subject]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            line = json.dumps(asdict(item), ensure_ascii=False)
            handle.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
    manifest = {
        "source": args.source,
        "n_items": len(items),
        "n_subjects": len(by_subject),
        "per_subject": {s: len(v) for s, v in sorted(by_subject.items())},
        "sha256": digest.hexdigest(),
        "file": out.name,
    }
    out.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_subject"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
