import argparse
import json
from collections import Counter
from pathlib import Path


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=4000)
    args = parser.parse_args()

    ids = set()
    text_hashes = set()
    langs = Counter()
    total = 0
    kept = 0
    duplicate_ids = 0
    duplicate_texts = 0
    empty = 0
    too_short = 0
    too_long = 0

    for line_no, row in iter_jsonl(args.corpus):
        total += 1
        doc_id = row.get("id")
        text = row.get("text")

        if not isinstance(doc_id, str) or not doc_id.strip():
            raise SystemExit(f"Missing/invalid id on line {line_no}")
        if doc_id in ids:
            duplicate_ids += 1
        ids.add(doc_id)

        if not isinstance(text, str) or not text.strip():
            empty += 1
            continue

        text = text.strip()
        if len(text) < args.min_chars:
            too_short += 1
            continue
        if len(text) > args.max_chars:
            too_long += 1
            continue

        text_key = hash(text)
        if text_key in text_hashes:
            duplicate_texts += 1
            continue
        text_hashes.add(text_key)

        langs[row.get("lang") or "unknown"] += 1
        kept += 1

    print(json.dumps({
        "total_rows": total,
        "kept_candidate_rows": kept,
        "duplicate_ids": duplicate_ids,
        "duplicate_texts": duplicate_texts,
        "empty_text": empty,
        "too_short": too_short,
        "too_long": too_long,
        "languages_top20": langs.most_common(20),
    }, indent=2))


if __name__ == "__main__":
    main()
