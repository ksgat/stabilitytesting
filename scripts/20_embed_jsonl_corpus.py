import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from larp_index import HFMeanPoolEmbedder


def load_texts(path: Path, text_field: str, limit: int | None) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get(text_field) or "").strip()
            if not text:
                raise RuntimeError(f"Missing text field {text_field!r} at row {len(texts)}")
            texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = load_texts(args.corpus, args.text_field, args.limit)
    if not texts:
        raise RuntimeError(f"No texts loaded from {args.corpus}")
    print(f"Loaded {len(texts)} texts from {args.corpus}", flush=True)

    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.out_embeddings.with_suffix(args.out_embeddings.suffix + ".progress")
    embedder = HFMeanPoolEmbedder(args.model_name, max_tokens=args.max_tokens)

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
