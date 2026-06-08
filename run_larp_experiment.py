import argparse
import csv
import hashlib
import json
import math
import os
import random
import textwrap
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer


DATASET_URL = "https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Text/code-to-text/dataset.zip"
HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "google/code_x_glue_ct_code_to_text"
DEFAULT_MODELS = [
    "microsoft/codebert-base",
    "Salesforce/codet5-small",
    "microsoft/unixcoder-base",
]
GENERAL_SENTENCE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "intfloat/e5-small-v2",
    "BAAI/bge-small-en-v1.5",
]
LANGUAGE_ORDER = ["go", "java", "javascript", "php", "python", "ruby"]
TEXT_FIELD_CANDIDATES = ("text", "sentence", "code", "content", "docstring", "description")


class CodeT5SmallTokenizer:
    def __init__(self, model_name: str):
        snapshot = Path(snapshot_download(model_name, local_files_only=True))
        self.tokenizer = ByteLevelBPETokenizer(
            str(snapshot / "vocab.json"),
            str(snapshot / "merges.txt"),
        )
        self.bos_id = 1
        self.pad_id = 0
        self.eos_id = 2

    def __call__(
        self,
        texts: list[str],
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        encoded = []
        content_limit = max(1, max_length - 2)
        for text in texts:
            ids = self.tokenizer.encode(text).ids
            if truncation:
                ids = ids[:content_limit]
            encoded.append([self.bos_id, *ids, self.eos_id])

        width = max(len(ids) for ids in encoded) if padding else None
        input_ids = []
        attention = []
        for ids in encoded:
            if padding:
                pad = width - len(ids)
                input_ids.append(ids + [self.pad_id] * pad)
                attention.append([1] * len(ids) + [0] * pad)
            else:
                input_ids.append(ids)
                attention.append([1] * len(ids))

        if return_tensors != "pt":
            raise ValueError("CodeT5SmallTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "data": root / "data",
        "external": root / "data" / "external",
        "processed": root / "data" / "processed",
        "embeddings": root / "outputs" / "embeddings",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def download_file(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing download: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as response:
        total = int(response.headers.get("content-length", "0"))
        done = 0
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {done / total:5.1%}", end="\r")
        tmp.replace(dest)
    print(f"Downloaded {dest.stat().st_size / (1024 * 1024):.1f} MB")


def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists():
        return
    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def iter_jsonl_files(extract_dir: Path):
    for path in sorted(extract_dir.rglob("*.jsonl")):
        lower = path.as_posix().lower()
        lang = next((name for name in LANGUAGE_ORDER if f"/{name}/" in lower or f"\\{name}\\" in lower), None)
        if lang:
            yield lang, path


def fetch_hf_rows(language: str, offset: int, length: int) -> list[dict[str, object]]:
    query = urlencode(
        {
            "dataset": HF_DATASET,
            "config": language,
            "split": "train",
            "offset": offset,
            "length": length,
        }
    )
    url = f"{HF_ROWS_URL}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [item["row"] for item in payload.get("rows", [])]


def fetch_corpus_from_hf(paths: dict[str, Path], n_docs: int, seed: int, max_chars: int) -> list[dict[str, str]]:
    cache_path = paths["processed"] / f"hf_{HF_DATASET.replace('/', '_')}_n{n_docs}_seed{seed}.jsonl"
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) >= n_docs:
            print(f"Using cached HF corpus: {cache_path}")
            return rows[:n_docs]

    rng = random.Random(seed)
    target_per_lang = max(1, math.ceil(n_docs / len(LANGUAGE_ORDER)))
    corpus: list[dict[str, str]] = []

    for lang in LANGUAGE_ORDER:
        lang_rows: list[dict[str, str]] = []
        offset = rng.randrange(0, 500)
        page = 100
        attempts = 0
        while len(lang_rows) < target_per_lang and attempts < 40:
            rows = fetch_hf_rows(lang, offset, page)
            if not rows:
                offset = 0
                rows = fetch_hf_rows(lang, offset, page)
                if not rows:
                    break
            for row in rows:
                code = (row.get("code") or row.get("original_string") or "").strip()
                if len(code) < 80:
                    continue
                lang_rows.append(
                    {
                        "language": lang,
                        "repo": row.get("repo", ""),
                        "path": row.get("path", ""),
                        "code": code[:max_chars],
                        "text": code[:max_chars],
                    }
                )
                if len(lang_rows) >= target_per_lang:
                    break
            offset += page
            attempts += 1
        print(f"HF rows: {lang} -> {len(lang_rows)} snippets")
        corpus.extend(lang_rows[:target_per_lang])

    rng.shuffle(corpus)
    corpus = corpus[:n_docs]
    if len(corpus) < n_docs:
        raise RuntimeError(f"Only found {len(corpus)} usable HF snippets, wanted {n_docs}")

    with cache_path.open("w", encoding="utf-8") as fh:
        for row in corpus:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"HF corpus: {len(corpus)} snippets -> {cache_path}")
    return corpus


def infer_text(row: dict[str, object], preferred_field: str | None = None) -> tuple[str, str | None]:
    fields = [preferred_field] if preferred_field else []
    fields.extend(field for field in TEXT_FIELD_CANDIDATES if field not in fields)
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return "", None


def generated_id(row: dict[str, object], line_no: int, text: str) -> str:
    value = row.get("id") or row.get("doc_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    path = row.get("path")
    if isinstance(path, str) and path.strip():
        return f"{path.strip()}-{digest}"
    return f"generated-{line_no}-{digest}"


def load_jsonl_corpus(
    corpus_path: Path,
    n_docs: int,
    seed: int,
    max_chars: int,
    min_chars: int,
    text_field: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_text = set()

    with corpus_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            text, used_field = infer_text(row, text_field)
            if len(text) < min_chars:
                continue
            text = text[:max_chars]
            if text in seen_text:
                continue
            seen_text.add(text)

            rows.append(
                {
                    "id": generated_id(row, line_no, text),
                    "language": str(row.get("language") or row.get("lang") or row.get("domain") or "unknown"),
                    "repo": str(row.get("repo") or ""),
                    "path": str(row.get("path") or ""),
                    "text": text,
                    "text_field": used_field or "",
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_docs]
    if len(rows) < n_docs:
        print(f"warning: requested {n_docs} rows from {corpus_path}, loaded {len(rows)}")
    print(f"Loaded external corpus: {len(rows)} rows from {corpus_path}")
    return rows


def load_corpus(
    paths: dict[str, Path],
    n_docs: int,
    seed: int,
    max_chars: int,
    min_chars: int = 80,
    corpus_path: Path | None = None,
    text_field: str | None = None,
) -> list[dict[str, str]]:
    if corpus_path:
        return load_jsonl_corpus(corpus_path, n_docs, seed, max_chars, min_chars, text_field)

    zip_path = paths["external"] / "codexglue_code_to_text_dataset.zip"
    extract_dir = paths["external"] / "codexglue_code_to_text"
    download_file(DATASET_URL, zip_path)
    extract_dataset(zip_path, extract_dir)

    rng = random.Random(seed)
    target_per_lang = max(1, math.ceil(n_docs / len(LANGUAGE_ORDER)))
    by_lang: dict[str, list[dict[str, str]]] = {lang: [] for lang in LANGUAGE_ORDER}

    for lang, path in iter_jsonl_files(extract_dir):
        if len(by_lang[lang]) >= target_per_lang * 3:
            continue
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if len(by_lang[lang]) >= target_per_lang * 3:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = row.get("code") or row.get("original_string") or ""
                code = code.strip()
                if len(code) < 80:
                    continue
                by_lang[lang].append(
                    {
                        "language": lang,
                        "repo": row.get("repo", ""),
                        "path": row.get("path", ""),
                        "code": code[:max_chars],
                        "text": code[:max_chars],
                    }
                )

    corpus: list[dict[str, str]] = []
    for lang in LANGUAGE_ORDER:
        rows = by_lang[lang]
        rng.shuffle(rows)
        corpus.extend(rows[:target_per_lang])

    rng.shuffle(corpus)
    corpus = corpus[:n_docs]
    if len(corpus) < n_docs:
        print(
            "CodeXGLUE zip did not include materialized code rows; "
            "falling back to Hugging Face dataset viewer rows."
        )
        return fetch_corpus_from_hf(paths, n_docs, seed, max_chars)

    out_path = paths["processed"] / f"codexglue_sample_n{n_docs}_seed{seed}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in corpus:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Corpus: {len(corpus)} snippets -> {out_path}")
    return corpus


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    z = l2_normalize(embeddings.astype(np.float32))
    sim = np.clip(z @ z.T, -1.0, 1.0)
    return 1.0 - sim


def anchor_relative_signature(embeddings: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    docs = l2_normalize(embeddings.astype(np.float32))
    anchors = docs[anchor_indices]
    return docs @ anchors.T


def mrr_preservation(dist_a: np.ndarray, dist_b: np.ndarray, k: int = 10) -> float:
    n = dist_a.shape[0]
    reciprocal_ranks = []
    for i in range(n):
        top_a = set(np.argsort(dist_a[i])[1 : k + 1])
        ranked_b = np.argsort(dist_b[i])[1 : k + 1]
        rank = next((r for r, idx in enumerate(ranked_b, start=1) if idx in top_a), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(reciprocal_ranks))


def top1_overlap(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    nn_a = np.argsort(dist_a, axis=1)[:, 1]
    nn_b = np.argsort(dist_b, axis=1)[:, 1]
    return float(np.mean(nn_a == nn_b))


def distance_spearman(dist_a: np.ndarray, dist_b: np.ndarray, max_pairs: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = dist_a.shape[0]
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        tri = np.triu_indices(n, k=1)
        a = dist_a[tri]
        b = dist_b[tri]
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = np.where(j >= i, j + 1, j)
        a = dist_a[i, j]
        b = dist_b[i, j]
    return float(spearmanr(a, b).statistic)


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def forward_embeddings(model, tokenizer, texts: list[str], device: str, max_tokens: int) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        if getattr(model.config, "is_encoder_decoder", False) and hasattr(model, "encoder"):
            outputs = model.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        else:
            try:
                outputs = model(**batch)
            except TypeError:
                outputs = model(batch["input_ids"])

    if isinstance(outputs, torch.Tensor):
        emb = outputs
    elif isinstance(outputs, (tuple, list)) and isinstance(outputs[0], torch.Tensor):
        first = outputs[0]
        emb = first if first.ndim == 2 else mean_pool(first, batch["attention_mask"])
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output
    elif hasattr(outputs, "last_hidden_state"):
        emb = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
    else:
        raise RuntimeError(f"Unsupported model output type: {type(outputs)}")

    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return torch.nn.functional.normalize(emb.float(), p=2, dim=1).cpu()


def embed_corpus(
    model_name: str,
    texts: list[str],
    paths: dict[str, Path],
    batch_size: int,
    max_tokens: int,
    device: str,
    allow_remote_code: bool,
) -> np.ndarray:
    out_path = paths["embeddings"] / f"{model_slug(model_name)}_n{len(texts)}_tok{max_tokens}.npy"
    if out_path.exists():
        print(f"Loading cached embeddings for {model_name}: {out_path}")
        return np.load(out_path)

    if "codet5p-110m-embedding" in model_name.lower() and not allow_remote_code:
        raise RuntimeError(
            "Salesforce/codet5p-110m-embedding requires trust_remote_code=True; "
            "rerun with --allow-remote-code only if you accept executing third-party HF model code."
        )

    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=allow_remote_code)
    except TypeError:
        if model_name == "Salesforce/codet5-small":
            tokenizer = CodeT5SmallTokenizer(model_name)
        else:
            raise
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=allow_remote_code)
    # The current CodeT5+ embedding remote config omits this T5Stack field under
    # transformers 5.5.x, while the remote model code still reads it.
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    model = AutoModel.from_pretrained(model_name, trust_remote_code=allow_remote_code, config=config)
    model.to(device)
    model.eval()

    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        if model_name.startswith("intfloat/e5-"):
            batch_texts = [f"passage: {text}" for text in batch_texts]
        emb = forward_embeddings(model, tokenizer, batch_texts, device, max_tokens)
        all_embeddings.append(emb.numpy())
        print(f"  {model_name}: {min(start + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    np.save(out_path, embeddings)
    print(f"Saved embeddings: {out_path}")
    return embeddings


def experiment_cross_model(
    embeddings: dict[str, np.ndarray],
    anchor_indices: np.ndarray,
    max_pairs: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    prepared = {}
    for name, emb in embeddings.items():
        sig = anchor_relative_signature(emb, anchor_indices)
        prepared[name] = {
            "abs_dist": cosine_distance_matrix(emb),
            "rel_dist": cosine_distance_matrix(sig),
        }

    rows = []
    names = list(embeddings)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            abs_mrr = mrr_preservation(prepared[a]["abs_dist"], prepared[b]["abs_dist"])
            rel_mrr = mrr_preservation(prepared[a]["rel_dist"], prepared[b]["rel_dist"])
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "absolute_mrr": abs_mrr,
                    "relative_mrr": rel_mrr,
                    "delta_mrr": rel_mrr - abs_mrr,
                    "absolute_spearman": distance_spearman(prepared[a]["abs_dist"], prepared[b]["abs_dist"], max_pairs, seed),
                    "relative_spearman": distance_spearman(prepared[a]["rel_dist"], prepared[b]["rel_dist"], max_pairs, seed),
                    "absolute_top1_overlap": top1_overlap(prepared[a]["abs_dist"], prepared[b]["abs_dist"]),
                    "relative_top1_overlap": top1_overlap(prepared[a]["rel_dist"], prepared[b]["rel_dist"]),
                }
            )
    return rows, prepared


def experiment_anchor_counts(
    embeddings: np.ndarray,
    anchor_pool: np.ndarray,
    counts: list[int],
) -> list[dict[str, object]]:
    abs_dist = cosine_distance_matrix(embeddings)
    rows = []
    for count in counts:
        use = anchor_pool[:count]
        sig = anchor_relative_signature(embeddings, use)
        sig_dist = cosine_distance_matrix(sig)
        rows.append(
            {
                "anchor_count": count,
                "mrr_vs_absolute": mrr_preservation(abs_dist, sig_dist),
                "spearman_vs_absolute": distance_spearman(abs_dist, sig_dist, 200_000, 17),
                "top1_overlap_vs_absolute": top1_overlap(abs_dist, sig_dist),
            }
        )
    return rows


def experiment_perturbation(
    embeddings: np.ndarray,
    baseline_n: int,
    anchor_pool: np.ndarray,
    expansion_fracs: list[float],
    anchor_count: int,
) -> list[dict[str, object]]:
    baseline_anchor_indices = anchor_pool[anchor_pool < baseline_n][:anchor_count]
    if len(baseline_anchor_indices) < anchor_count:
        baseline_anchor_indices = np.arange(min(anchor_count, baseline_n))

    baseline_sigs = anchor_relative_signature(embeddings[:baseline_n], baseline_anchor_indices)
    rows = []
    for frac in expansion_fracs:
        full_n = min(len(embeddings), baseline_n + int(round(baseline_n * frac)))
        # Stable anchors mean the existing anchor set stays fixed while new docs arrive.
        stable_sigs = anchor_relative_signature(embeddings[:full_n], baseline_anchor_indices)[:baseline_n]
        stable_sim = np.mean(np.sum(l2_normalize(baseline_sigs) * l2_normalize(stable_sigs), axis=1))

        # Refit anchors simulates the tempting but expensive choice of changing anchors with the corpus.
        refit_pool = anchor_pool[anchor_pool < full_n]
        refit_anchor_indices = refit_pool[: len(baseline_anchor_indices)]
        refit_sigs = anchor_relative_signature(embeddings[:full_n], refit_anchor_indices)[:baseline_n]
        refit_sim = np.mean(np.sum(l2_normalize(baseline_sigs) * l2_normalize(refit_sigs), axis=1))

        rows.append(
            {
                "expansion_fraction": frac,
                "baseline_docs": baseline_n,
                "full_docs": full_n,
                "stable_anchor_drift": 1.0 - float(stable_sim),
                "refit_anchor_drift": 1.0 - float(refit_sim),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def plot_cross_model(rows: list[dict[str, object]], path: Path) -> None:
    labels = [f"{r['model_a'].split('/')[-1]}\nvs\n{r['model_b'].split('/')[-1]}" for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    fig_width = max(11, len(rows) * 1.15)
    fig, ax = plt.subplots(figsize=(fig_width, 6.8))
    ax.bar(x - width / 2, [r["absolute_mrr"] for r in rows], width, label="Absolute", color="#4C78A8")
    ax.bar(x + width / 2, [r["relative_mrr"] for r in rows], width, label="Anchor-relative", color="#F58518")
    ax.set_ylabel("MRR@10 preservation")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-model nearest-neighbor preservation")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_anchor_counts(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot([r["anchor_count"] for r in rows], [r["mrr_vs_absolute"] for r in rows], marker="o", color="#54A24B")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Anchor count")
    ax.set_ylabel("MRR@10 vs raw embedding neighbors")
    ax.set_title("Anchor count vs retrieval preservation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_perturbation(rows: list[dict[str, object]], path: Path) -> None:
    x = [100 * r["expansion_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, [r["stable_anchor_drift"] for r in rows], marker="o", label="Fixed anchors", color="#4C78A8")
    ax.plot(x, [r["refit_anchor_drift"] for r in rows], marker="o", label="Refit anchors", color="#E45756")
    ax.set_xlabel("Corpus expansion")
    ax.set_ylabel("Signature drift, lower is better")
    ax.set_title("Existing-document signature drift under corpus growth")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(path: Path, args, corpus, cross_rows, anchor_rows, perturb_rows, models_completed) -> None:
    by_lang = {}
    for row in corpus:
        lang = row.get("language") or "unknown"
        by_lang[lang] = by_lang.get(lang, 0) + 1

    lines = [
        "# LARP Experiment Results",
        "",
        f"- Run: {args.run_name}",
        f"- Docs sampled: {len(corpus)}",
        f"- Language mix: {by_lang}",
        f"- Models completed: {models_completed}",
        f"- Anchor count: {args.anchor_count}",
        f"- Max tokens/snippet: {args.max_tokens}",
        "",
        "## Experiment 1: Cross-model Preservation",
        "",
    ]
    if cross_rows:
        lines.append("| Model A | Model B | Abs MRR | Rel MRR | Delta | Abs rho | Rel rho |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in cross_rows:
            lines.append(
                f"| {r['model_a'].split('/')[-1]} | {r['model_b'].split('/')[-1]} | "
                f"{r['absolute_mrr']:.4f} | {r['relative_mrr']:.4f} | {r['delta_mrr']:+.4f} | "
                f"{r['absolute_spearman']:.4f} | {r['relative_spearman']:.4f} |"
            )
    else:
        lines.append("Not enough completed models for cross-model comparison.")

    lines.extend(["", "## Experiment 2: Corpus Perturbation", ""])
    lines.append("| Expansion | Fixed-anchor drift | Refit-anchor drift |")
    lines.append("|---:|---:|---:|")
    for r in perturb_rows:
        lines.append(f"| {100 * r['expansion_fraction']:.0f}% | {r['stable_anchor_drift']:.6f} | {r['refit_anchor_drift']:.6f} |")

    lines.extend(["", "## Experiment 3: Anchor Count", ""])
    lines.append("| Anchors | MRR | Spearman | Top-1 overlap |")
    lines.append("|---:|---:|---:|---:|")
    for r in anchor_rows:
        lines.append(
            f"| {r['anchor_count']} | {r['mrr_vs_absolute']:.4f} | "
            f"{r['spearman_vs_absolute']:.4f} | {r['top1_overlap_vs_absolute']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            textwrap.dedent(
                """
                Fixed-anchor perturbation is algebraically stable in this setup because transformer
                embeddings are computed independently per snippet and the anchors are unchanged.
                The refit-anchor curve is the meaningful warning signal: if anchors are reselected
                when the corpus grows, existing signatures move and a rebuild-like update appears.
                """
            ).strip(),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test anchor-relative code embedding stability.")
    parser.add_argument("--n-docs", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--model-suite", choices=["code", "general-sentence"], default=None)
    parser.add_argument("--run-name", default="code_models")
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-spearman-pairs", type=int, default=200_000)
    parser.add_argument("--allow-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_suite == "general-sentence":
        args.models = GENERAL_SENTENCE_MODELS
        if args.run_name == "code_models":
            args.run_name = "general_sentence_models"
    elif args.model_suite == "code" and args.run_name == "code_models":
        args.models = DEFAULT_MODELS

    root = Path.cwd()
    paths = ensure_dirs(root)
    rng = np.random.default_rng(args.seed)

    corpus = load_corpus(
        paths,
        args.n_docs,
        args.seed,
        args.max_chars,
        args.min_chars,
        args.corpus_path,
        args.text_field,
    )
    texts = [row.get("text") or row.get("code") or "" for row in corpus]
    if len(texts) < 2:
        raise RuntimeError("Need at least two usable corpus rows to run stability tests.")

    n_loaded = len(texts)
    anchor_count = min(args.anchor_count, max(4, n_loaded // 2))
    anchor_pool = rng.permutation(n_loaded)
    anchor_indices = anchor_pool[:anchor_count]

    embeddings = {}
    failures = {}
    for model_name in args.models:
        try:
            embeddings[model_name] = embed_corpus(
                model_name,
                texts,
                paths,
                args.batch_size,
                args.max_tokens,
                args.device,
                args.allow_remote_code,
            )
        except Exception as exc:
            failures[model_name] = repr(exc)
            print(f"FAILED {model_name}: {exc}")

    if not embeddings:
        raise RuntimeError(f"No model embeddings completed. Failures: {failures}")

    cross_rows, _ = experiment_cross_model(embeddings, anchor_indices, args.max_spearman_pairs, args.seed)
    primary_model = next(iter(embeddings))
    counts = [c for c in [8, 16, 32, 64, 128, 256, 512] if c <= n_loaded // 2]
    anchor_rows = experiment_anchor_counts(embeddings[primary_model], anchor_pool, counts)
    baseline_n = max(2, min(int(n_loaded * 0.5), n_loaded - 1))
    perturb_rows = experiment_perturbation(
        embeddings[primary_model],
        baseline_n,
        anchor_pool,
        [0.10, 0.25, 0.50, 1.00],
        min(anchor_count, baseline_n // 2),
    )

    table_prefix = f"{args.run_name}_"
    figure_prefix = f"{args.run_name}_"
    write_csv(paths["tables"] / f"{table_prefix}cross_model_preservation.csv", cross_rows)
    write_csv(paths["tables"] / f"{table_prefix}anchor_count_curve.csv", anchor_rows)
    write_csv(paths["tables"] / f"{table_prefix}corpus_perturbation.csv", perturb_rows)
    if cross_rows:
        plot_cross_model(cross_rows, paths["figures"] / f"{figure_prefix}cross_model_mrr.png")
    plot_anchor_counts(anchor_rows, paths["figures"] / f"{figure_prefix}anchor_count_curve.png")
    plot_perturbation(perturb_rows, paths["figures"] / f"{figure_prefix}corpus_perturbation.png")
    write_report(
        root / f"larp_results_{args.run_name}.md",
        args,
        corpus,
        cross_rows,
        anchor_rows,
        perturb_rows,
        list(embeddings),
    )

    if failures:
        print("Model failures:")
        for name, err in failures.items():
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
