import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_TEXT_FIELDS = ("text", "sentence", "code", "content", "docstring", "description")


def slug(name):
    return name.replace("/", "__")


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
        return value.strip()
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    path = row.get("path")
    if isinstance(path, str) and path.strip():
        return f"{path.strip()}-{digest}"
    return f"generated-{line_no}-{digest}"


def load_corpus(path, n_docs, text_field=None):
    rows = []
    seen_text = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text, field_used = extract_text(row, text_field)
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            rows.append({
                "id": stable_id(row, line_no, text),
                "text": text,
                "path": row.get("path"),
                "lang": row.get("lang") or row.get("language") or row.get("domain"),
                "repo": row.get("repo"),
                "text_field": field_used,
            })
            if len(rows) >= n_docs:
                break
    if len(rows) < n_docs:
        print(f"warning: requested {n_docs} docs but only loaded {len(rows)}")
    return rows


def resolve_device(config_device):
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config_device


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_texts(model_name, texts, batch_size, max_length, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc=model_name):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = model(**encoded)

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            pooled = output.pooler_output
        else:
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])

        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(config["run_dir"])
    embed_dir = run_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    rows = load_corpus(config["corpus_path"], config["n_docs"], config.get("text_field"))
    metadata_path = run_dir / "corpus_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    texts = [row["text"] for row in rows]
    device = resolve_device(config["device"])
    print(f"device={device} docs={len(texts)}")

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        out_path = embed_dir / f"{slug(model_name)}.npy"
        if out_path.exists():
            print(f"skip existing {out_path}")
            continue
        vectors = embed_texts(
            model_name,
            texts,
            config["batch_size"],
            config["max_length"],
            device,
        )
        np.save(out_path, vectors)
        print(f"saved {out_path} shape={vectors.shape}")


if __name__ == "__main__":
    main()
