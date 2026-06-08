import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


DEFAULT_TEXT_FIELDS = ("text", "sentence", "code", "content", "docstring", "description")


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


def extract_text(row, preferred_field=None):
    fields = [preferred_field] if preferred_field else []
    fields.extend(field for field in DEFAULT_TEXT_FIELDS if field not in fields)
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", None


def stable_id(row, line_no, text):
    value = row.get("id") or row.get("doc_id")
    if isinstance(value, str) and value.strip():
        return value.strip(), False
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    path = row.get("path")
    if isinstance(path, str) and path.strip():
        return f"{path.strip()}-{digest}", True
    return f"generated-{line_no}-{digest}", True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--text-field", default=None)
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
    generated_ids = 0
    fields_used = Counter()

    for line_no, row in iter_jsonl(args.corpus):
        total += 1
        text, field_used = extract_text(row, args.text_field)
        doc_id, was_generated = stable_id(row, line_no, text)
        generated_ids += int(was_generated)

        if doc_id in ids:
            duplicate_ids += 1
        ids.add(doc_id)

        if not text:
            empty += 1
            continue

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

        fields_used[field_used or "unknown"] += 1
        langs[row.get("lang") or row.get("language") or row.get("domain") or "unknown"] += 1
        kept += 1

    print(json.dumps({
        "total_rows": total,
        "kept_candidate_rows": kept,
        "duplicate_ids": duplicate_ids,
        "generated_ids": generated_ids,
        "duplicate_texts": duplicate_texts,
        "empty_text": empty,
        "too_short": too_short,
        "too_long": too_long,
        "text_fields_used": fields_used.most_common(),
        "languages_top20": langs.most_common(20),
    }, indent=2))


if __name__ == "__main__":
    main()
