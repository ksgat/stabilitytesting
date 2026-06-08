import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def slug(name):
    return name.replace("/", "__")


def normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def choose_random_anchors(n_docs, count, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_docs, size=count, replace=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(config["run_dir"])
    sig_dir = run_dir / "signatures"
    sig_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((run_dir / "corpus_metadata.json").read_text(encoding="utf-8"))
    n_docs = len(metadata)
    anchor_count = min(config["anchor"]["count"], n_docs)
    anchor_idx = choose_random_anchors(n_docs, anchor_count, config["seed"])

    np.save(sig_dir / "anchor_indices.npy", anchor_idx)
    (sig_dir / "anchor_ids.json").write_text(
        json.dumps([metadata[i]["id"] for i in anchor_idx], indent=2),
        encoding="utf-8",
    )

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        emb_path = run_dir / "embeddings" / f"{slug(model_name)}.npy"
        embeddings = normalize(np.load(emb_path))
        anchors = embeddings[anchor_idx]
        signatures = embeddings @ anchors.T
        signatures = normalize(signatures.astype("float32"))
        out_path = sig_dir / f"{slug(model_name)}.npy"
        np.save(out_path, signatures)
        print(f"saved {out_path} shape={signatures.shape}")


if __name__ == "__main__":
    main()
