import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path, seed: int) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            row["seed"] = seed
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="seed=path/to/learned_anchor_selection.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/learned_anchor_seed_report"))
    parser.add_argument("--pool", type=int, default=250)
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        seed_s, path_s = run.split("=", 1)
        rows.extend(read_rows(Path(path_s), int(seed_s)))

    selected = [r for r in rows if int(r["pool_size"]) == args.pool]
    summary = []
    for strategy in sorted({r["strategy"] for r in selected}):
        vals = np.array([float(r["mean_recall"]) for r in selected if r["strategy"] == strategy], dtype=np.float64)
        summary.append(
            {
                "strategy": strategy,
                "pool_size": args.pool,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=0)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "runs": len(vals),
            }
        )

    by_model = []
    for model in sorted({r["model"] for r in selected}):
        base = np.array(
            [float(r["mean_recall"]) for r in selected if r["model"] == model and r["strategy"] == "farthest_row_zscore"],
            dtype=np.float64,
        )
        learned = np.array(
            [float(r["mean_recall"]) for r in selected if r["model"] == model and r["strategy"] == "learned_top256"],
            dtype=np.float64,
        )
        by_model.append(
            {
                "model": model,
                "pool_size": args.pool,
                "farthest_mean": float(base.mean()),
                "learned_mean": float(learned.mean()),
                "delta_mean": float((learned - base).mean()),
                "delta_min": float((learned - base).min()),
                "delta_max": float((learned - base).max()),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "learned_anchor_seed_summary.csv", summary)
    write_csv(args.out_dir / "learned_anchor_seed_by_model.csv", by_model)

    lines = [
        "# Learned Anchor Seed Generalization",
        "",
        f"Pool evaluated: {args.pool}",
        "",
        "## Strategy Summary",
        "",
        *markdown_table(
            ["strategy", "mean", "std", "min", "max", "rows"],
            [
                [
                    r["strategy"],
                    f"{r['mean']:.4f}",
                    f"{r['std']:.4f}",
                    f"{r['min']:.4f}",
                    f"{r['max']:.4f}",
                    r["runs"],
                ]
                for r in summary
            ],
        ),
        "",
        "## Learned-vs-Farthest by Model",
        "",
        *markdown_table(
            ["model", "farthest mean", "learned mean", "delta mean", "delta min", "delta max"],
            [
                [
                    r["model"],
                    f"{r['farthest_mean']:.4f}",
                    f"{r['learned_mean']:.4f}",
                    f"{r['delta_mean']:+.4f}",
                    f"{r['delta_min']:+.4f}",
                    f"{r['delta_max']:+.4f}",
                ]
                for r in by_model
            ],
        ),
        "",
        "## Finding",
        "",
        "Across the supplied seeds, learned anchor selection should be considered robust only if its delta over farthest anchors is positive for every model and the variance is small relative to the gain.",
    ]
    (args.out_dir / "learned_anchor_seed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_dir / 'learned_anchor_seed_report.md'}")


if __name__ == "__main__":
    main()
