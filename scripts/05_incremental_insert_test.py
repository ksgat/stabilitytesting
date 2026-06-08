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


def neighbor_change_due_to_insert(full_dist, base_n, inserted_n, k):
    base_only = full_dist[:base_n, :base_n]
    expanded = full_dist[:base_n, :base_n + inserted_n]

    base_topk = metrics.topk_indices(base_only, k)
    expanded_ranked = np.argsort(expanded, axis=1)

    changed = 0
    inserted_hits = []
    retained_overlap = []

    for i in range(base_n):
        before = set(base_topk[i].tolist())
        after = []
        for idx in expanded_ranked[i]:
            if idx == i:
                continue
            after.append(int(idx))
            if len(after) == k:
                break

        after_set = set(after)
        if before != after_set:
            changed += 1
        inserted_in_topk = [idx for idx in after if idx >= base_n]
        inserted_hits.append(len(inserted_in_topk))
        retained_overlap.append(len(before & after_set) / k)

    return {
        "changed_existing_doc_fraction": changed / base_n,
        "mean_inserted_docs_in_existing_topk": float(np.mean(inserted_hits)),
        "mean_retained_topk_overlap": float(np.mean(retained_overlap)),
    }


def signature_reselection_instability(embeddings, base_n, inserted_n, anchor_count, seed, k):
    rng = np.random.default_rng(seed)
    anchor_count = min(anchor_count, base_n, base_n + inserted_n)
    base_anchor_idx = np.sort(rng.choice(base_n, size=anchor_count, replace=False))
    full_anchor_idx = np.sort(rng.choice(base_n + inserted_n, size=anchor_count, replace=False))

    base_embeddings = embeddings[:base_n]
    full_embeddings = embeddings[:base_n + inserted_n]

    base_sigs = normalize((base_embeddings @ embeddings[base_anchor_idx].T).astype("float32"))
    reselected_sigs = normalize((base_embeddings @ full_embeddings[full_anchor_idx].T).astype("float32"))

    base_sig_dist = metrics.cosine_distance_matrix(base_sigs)
    reselected_sig_dist = metrics.cosine_distance_matrix(reselected_sigs)

    return {
        "mrr_fixed_base_anchors_to_reselected_anchors": metrics.mrr_agreement(base_sig_dist, reselected_sig_dist, k),
        "topk_overlap_fixed_base_anchors_to_reselected_anchors": metrics.topk_overlap(base_sig_dist, reselected_sig_dist, k),
    }


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
    n_docs = embeddings.shape[0]
    base_n = int(n_docs * config["incremental"]["base_fraction"])
    k = config["metrics"]["k"]
    anchor_count = config["anchor"]["count"]

    rows = []
    for fraction in config["incremental"]["insert_fractions"]:
        inserted_n = min(int(n_docs * fraction), n_docs - base_n)
        if inserted_n <= 0:
            continue
        subset = embeddings[:base_n + inserted_n]
        full_dist = metrics.cosine_distance_matrix(subset)

        insert_metrics = neighbor_change_due_to_insert(full_dist, base_n, inserted_n, k)
        anchor_metrics = signature_reselection_instability(
            embeddings,
            base_n,
            inserted_n,
            anchor_count,
            config["seed"],
            k,
        )
        rows.append({
            "model": model_name,
            "base_docs": base_n,
            "inserted_docs": inserted_n,
            "insert_fraction_of_full_corpus": fraction,
            **insert_metrics,
            **anchor_metrics,
        })

    out_path = metric_dir / "incremental_insert_test.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
