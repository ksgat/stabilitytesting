import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


DEFAULT_FILES = [
    "python/train-00000-of-00002.parquet",
    "python/train-00001-of-00002.parquet",
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def pick_code(row: dict[str, Any]) -> str:
    for key in ("code", "func_code_string", "original_string", "text"):
        text = normalize_text(row.get(key)).strip()
        if text:
            return text
    return ""


def pick_docstring(row: dict[str, Any]) -> str:
    for key in ("docstring", "func_documentation_string", "summary"):
        text = normalize_text(row.get(key)).strip()
        if text:
            return text
    return ""


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="google/code_x_glue_ct_code_to_text")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES)
    parser.add_argument("--download-dir", type=Path, default=Path("data/external/hf_code_x_glue_ct_code_to_text"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/hf_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/real_distinct_hf_code/data"))
    parser.add_argument("--limits", nargs="+", type=int, default=[100000, 200000])
    parser.add_argument("--min-code-chars", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = sorted(set(args.limits))
    max_limit = max(limits)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    local_files = []
    for filename in args.files:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=filename,
            local_dir=args.download_dir,
            cache_dir=args.cache_dir,
        )
        local_files.append(Path(local_path))
        print(f"Downloaded {filename} -> {local_path}", flush=True)

    writers = {
        limit: (args.out_dir / f"hf_code_x_glue_python_distinct_{limit}.jsonl").open("w", encoding="utf-8")
        for limit in limits
    }
    seen: set[str] = set()
    written = 0
    scanned = 0
    columns: list[str] | None = None
    try:
        for source_file in local_files:
            parquet = pq.ParquetFile(source_file)
            columns = columns or list(parquet.schema.names)
            for batch in parquet.iter_batches(batch_size=args.batch_size):
                rows = batch.to_pylist()
                for row in rows:
                    scanned += 1
                    code = pick_code(row)
                    if len(code) < args.min_code_chars:
                        continue
                    code_hash = stable_hash(code)
                    if code_hash in seen:
                        continue
                    seen.add(code_hash)
                    item = {
                        "id": f"hf_code_x_glue_python:{written}",
                        "source_dataset": args.repo_id,
                        "source_file": source_file.name,
                        "language": normalize_text(row.get("language") or "python"),
                        "repo": normalize_text(row.get("repo")),
                        "path": normalize_text(row.get("path")),
                        "url": normalize_text(row.get("url")),
                        "code_hash": code_hash,
                        "text": code,
                        "docstring": pick_docstring(row),
                    }
                    line = json.dumps(item, ensure_ascii=True) + "\n"
                    written += 1
                    for limit, fh in writers.items():
                        if written <= limit:
                            fh.write(line)
                    if written % 10000 == 0:
                        print(f"Distinct rows {written}/{max_limit} scanned={scanned}", flush=True)
                    if written >= max_limit:
                        break
                if written >= max_limit:
                    break
            if written >= max_limit:
                break
    finally:
        for fh in writers.values():
            fh.close()

    if written < max_limit:
        raise RuntimeError(f"Only wrote {written} distinct rows; wanted {max_limit}. Add more files.")

    manifest = {
        "repo_id": args.repo_id,
        "files": args.files,
        "local_files": [str(path) for path in local_files],
        "columns": columns or [],
        "limits": limits,
        "scanned_rows": scanned,
        "distinct_rows": written,
        "min_code_chars": args.min_code_chars,
        "seconds": time.perf_counter() - start,
        "outputs": {
            str(limit): str(args.out_dir / f"hf_code_x_glue_python_distinct_{limit}.jsonl")
            for limit in limits
        },
    }
    write_manifest(args.out_dir / "hf_code_x_glue_python_distinct_manifest.json", manifest)
    print(f"Wrote {written} distinct rows after scanning {scanned}", flush=True)
    print(f"Manifest: {args.out_dir / 'hf_code_x_glue_python_distinct_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
