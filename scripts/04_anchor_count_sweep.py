import argparse
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import yaml


metrics = SourceFileLoader(
    "cross_model_metrics",
    str(Path(__file__).with_name("03_cross_model_preservation.py")),
).load_module()


def slug(name):
    return name.replace("/", "__")


def normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(denom, 1e-12, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path(config["run_dir"])
    metric_dir = run_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["models"][0]["name"]
    embeddings = normalize(np.load(run_dir / "embeddings" / f"{slug(model_name)}.npy"))
    raw_dist = metrics.cosine_distance_matrix(embeddings)
    anchor_indices = np.load(run_dir / "signatures" / "anchor_indices.npy")

    rows = []
    for count in config["anchor"]["counts_for_sweep"]:
        count = min(count, len(anchor_indices))
        selected = anchor_indices[:count]
        signatures = normalize((embeddings @ embeddings[selected].T).astype("float32"))
        sig_dist = metrics.cosine_distance_matrix(signatures)
        rows.append({
            "model": model_name,
            "anchor_count": int(count),
            "mrr_raw_to_signature": metrics.mrr_agreement(raw_dist, sig_dist, config["metrics"]["k"]),
            "topk_overlap": metrics.topk_overlap(raw_dist, sig_dist, config["metrics"]["k"]),
            "sampled_spearman": metrics.sampled_spearman(
                raw_dist,
                sig_dist,
                config["metrics"]["spearman_sample_pairs"],
                config["seed"],
            ),
        })

    out_path = metric_dir / "anchor_count_sweep.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
