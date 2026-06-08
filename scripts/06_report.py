import argparse
import json
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/baseline")
    args = parser.parse_args()

    metric_dir = Path(args.run_dir) / "metrics"
    cross = load_json(metric_dir / "cross_model_preservation.json")
    anchors = load_json(metric_dir / "anchor_count_sweep.json")
    incremental = load_json(metric_dir / "incremental_insert_test.json")

    lines = []
    lines.append("# Stability Test Report")
    lines.append("")
    lines.append("## Cross-model preservation")
    lines.append("")
    lines.append("| Representation | Model A | Model B | MRR A->B | MRR B->A | Top-k overlap | Spearman |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in cross:
        lines.append(
            f"| {row['representation']} | {row['a']} | {row['b']} | "
            f"{row['mrr_a_to_b']:.4f} | {row['mrr_b_to_a']:.4f} | "
            f"{row['topk_overlap']:.4f} | {row['sampled_spearman']:.4f} |"
        )

    lines.append("")
    lines.append("## Anchor count sweep")
    lines.append("")
    lines.append("| Anchors | MRR raw->signature | Top-k overlap | Spearman |")
    lines.append("|---:|---:|---:|---:|")
    for row in anchors:
        lines.append(
            f"| {row['anchor_count']} | {row['mrr_raw_to_signature']:.4f} | "
            f"{row['topk_overlap']:.4f} | {row['sampled_spearman']:.4f} |"
        )

    lines.append("")
    lines.append("## Incremental insert test")
    lines.append("")
    lines.append("| Base docs | Inserted docs | Changed old-doc top-k | Inserted docs in old top-k | Retained old top-k overlap | Reselected-anchor MRR |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in incremental:
        lines.append(
            f"| {row['base_docs']} | {row['inserted_docs']} | "
            f"{row['changed_existing_doc_fraction']:.4f} | "
            f"{row['mean_inserted_docs_in_existing_topk']:.4f} | "
            f"{row['mean_retained_topk_overlap']:.4f} | "
            f"{row['mrr_fixed_base_anchors_to_reselected_anchors']:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation checklist")
    lines.append("")
    lines.append("- Relative signatures should beat raw cross-model neighbor preservation by a clear margin.")
    lines.append("- Anchor sweep should show a knee before 512 anchors. If not, signatures may not compress enough to matter.")
    lines.append("- Fixed-anchor signature drift under insertion is expected to be zero for deterministic embedding models; judge insertion by neighbor-rank locality instead.")
    lines.append("- Reselected-anchor instability tells you how costly it is to change anchor sets after the index exists.")

    out = Path(args.run_dir) / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
