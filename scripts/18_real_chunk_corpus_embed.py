import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_index import HFMeanPoolEmbedder


def chunk_text(text: str, chunk_chars: int, stride_chars: int, min_chars: int) -> list[str]:
    text = " ".join(text.split())
    if len(text) < min_chars:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_chars]
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if start + chunk_chars >= len(text):
            break
        start += stride_chars
    return chunks


def build_chunks(
    source_path: Path,
    out_path: Path,
    target_chunks: int,
    chunk_chars: int,
    stride_chars: int,
    min_chars: int,
) -> list[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        texts = []
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    texts.append(row["text"])
                    if len(texts) >= target_chunks:
                        break
        if len(texts) >= target_chunks:
            return texts[:target_chunks]

    texts = []
    with source_path.open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for source_idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            raw_text = row.get("text") or row.get("code") or ""
            chunks = chunk_text(raw_text, chunk_chars, stride_chars, min_chars)
            for chunk_idx, chunk in enumerate(chunks):
                item = {
                    "id": f"{source_idx}:{chunk_idx}",
                    "source_index": source_idx,
                    "chunk_index": chunk_idx,
                    "language": row.get("language", "unknown"),
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "text": chunk,
                }
                out.write(json.dumps(item, ensure_ascii=True) + "\n")
                texts.append(chunk)
                if len(texts) >= target_chunks:
                    return texts
    if len(texts) < target_chunks:
        raise RuntimeError(f"Only built {len(texts)} chunks; wanted {target_chunks}. Lower min/stride or add corpus.")
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-corpus", type=Path, default=Path("experiments/tenk_minilm_candidate/data/processed/hf_google_code_x_glue_ct_code_to_text_n10000_seed7.jsonl"))
    parser.add_argument("--out-corpus", type=Path, default=Path("experiments/real_100k_chunks/data/code_chunks_100k.jsonl"))
    parser.add_argument("--out-embeddings", type=Path, default=Path("experiments/real_100k_chunks/embeddings/code_chunks_100k_minilm.npy"))
    parser.add_argument("--target-chunks", type=int, default=100000)
    parser.add_argument("--chunk-chars", type=int, default=320)
    parser.add_argument("--stride-chars", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = build_chunks(
        args.source_corpus,
        args.out_corpus,
        args.target_chunks,
        args.chunk_chars,
        args.stride_chars,
        args.min_chars,
    )
    print(f"Chunks ready: {len(texts)} -> {args.out_corpus}", flush=True)
    if args.build_only:
        return
    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    embedder = HFMeanPoolEmbedder(args.model_name, max_tokens=args.max_tokens)
    progress_path = args.out_embeddings.with_suffix(args.out_embeddings.suffix + ".progress")
    start = 0
    mmap = None
    if args.resume and args.out_embeddings.exists() and progress_path.exists():
        start = int(progress_path.read_text(encoding="utf-8").strip() or "0")
        mmap = np.load(args.out_embeddings, mmap_mode="r+")
        if len(mmap) != len(texts):
            raise RuntimeError(f"Existing embedding count {len(mmap)} does not match text count {len(texts)}")
        print(f"Resuming embeddings at row {start}/{len(texts)}", flush=True)

    if mmap is None:
        probe = embedder.encode(texts[:1], batch_size=1)
        mmap = np.lib.format.open_memmap(
            args.out_embeddings,
            mode="w+",
            dtype=np.float32,
            shape=(len(texts), probe.shape[1]),
        )
        mmap[0:1] = probe.astype(np.float32)
        start = 1
        progress_path.write_text(str(start), encoding="utf-8")
        print(f"Created embedding memmap: {args.out_embeddings} shape={mmap.shape}", flush=True)

    for batch_start in range(start, len(texts), args.batch_size):
        batch_end = min(len(texts), batch_start + args.batch_size)
        batch = embedder.encode(texts[batch_start:batch_end], batch_size=args.batch_size)
        mmap[batch_start:batch_end] = batch.astype(np.float32)
        mmap.flush()
        progress_path.write_text(str(batch_end), encoding="utf-8")
        if batch_end == len(texts) or batch_end % max(args.batch_size * 10, 1) == 0:
            print(f"Embedded {batch_end}/{len(texts)}", flush=True)

    mmap.flush()
    progress_path.write_text(str(len(texts)), encoding="utf-8")
    print(f"Wrote embeddings: {args.out_embeddings} shape={mmap.shape}", flush=True)


if __name__ == "__main__":
    main()
