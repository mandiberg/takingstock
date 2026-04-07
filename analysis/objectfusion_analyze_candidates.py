#!/usr/bin/env python3
"""Rank ObjectFusion candidate runs from even-split scorecard exports.

Input CSV should come from section 10 of analysis/objectfusion_scorecard_queries.sql.
Optional visual CSV can add a manual review signal per candidate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "mean_cluster_dist": 0.40,
    "mixed_rate": 0.30,
    "hhi_size_concentration": 0.20,
    "largest_cluster_ratio": 0.10,
}


def minmax_normalize(series: pd.Series, lower_is_better: bool = True) -> pd.Series:
    """Normalize a numeric series to [0,1], with 1 as better."""
    vals = pd.to_numeric(series, errors="coerce")
    finite_mask = np.isfinite(vals)
    if finite_mask.sum() == 0:
        return pd.Series(np.nan, index=series.index)

    lo = vals[finite_mask].min()
    hi = vals[finite_mask].max()

    if np.isclose(hi, lo):
        base = pd.Series(1.0, index=series.index)
    else:
        scaled = (vals - lo) / (hi - lo)
        base = 1.0 - scaled if lower_is_better else scaled

    base[~finite_mask] = np.nan
    return base


def load_visual_scores(path: Path) -> pd.DataFrame:
    """Load optional manual visual scores.

    Expected columns:
      candidate_name, visual_score
    visual_score should be in [0, 1] or [0, 100].
    """
    visual = pd.read_csv(path)
    required = {"candidate_name", "visual_score"}
    missing = required - set(visual.columns)
    if missing:
        raise ValueError(
            f"Missing columns in visual CSV: {sorted(missing)}. "
            "Expected columns: candidate_name, visual_score"
        )

    visual = visual[["candidate_name", "visual_score"]].copy()
    visual["visual_score"] = pd.to_numeric(visual["visual_score"], errors="coerce")

    max_val = visual["visual_score"].max(skipna=True)
    if pd.notna(max_val) and max_val > 1.0:
        visual["visual_score"] = visual["visual_score"] / 100.0

    visual["visual_score"] = visual["visual_score"].clip(lower=0.0, upper=1.0)
    return visual


def build_scorecard(df: pd.DataFrame, baseline_name: str, visual_df: pd.DataFrame | None) -> pd.DataFrame:
    required = {
        "candidate_name",
        "images_n",
        "mean_cluster_dist",
        "mixed_rate",
        "hhi_size_concentration",
        "largest_cluster",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in metrics CSV: {sorted(missing)}. "
            "Export section 10 exactly from analysis/objectfusion_scorecard_queries.sql"
        )

    out = df.copy()
    out["largest_cluster_ratio"] = pd.to_numeric(out["largest_cluster"], errors="coerce") / pd.to_numeric(
        out["images_n"], errors="coerce"
    )

    for metric in DEFAULT_WEIGHTS:
        out[f"score_{metric}"] = minmax_normalize(out[metric], lower_is_better=True)

    if visual_df is not None:
        out = out.merge(visual_df, on="candidate_name", how="left")
        out["score_visual"] = out["visual_score"]
    else:
        out["score_visual"] = np.nan

    visual_weight = 0.0 if out["score_visual"].isna().all() else 0.15
    metric_weight_scale = 1.0 - visual_weight

    weighted_metric_scores = pd.Series(0.0, index=out.index)
    for metric, weight in DEFAULT_WEIGHTS.items():
        weighted_metric_scores += out[f"score_{metric}"].fillna(0.0) * (weight * metric_weight_scale)

    out["composite_score"] = weighted_metric_scores + out["score_visual"].fillna(0.0) * visual_weight

    baseline_row = out[out["candidate_name"] == baseline_name]
    if baseline_row.empty:
        raise ValueError(
            f"Baseline candidate '{baseline_name}' not found in metrics CSV. "
            "Use --baseline to match your baseline candidate_name."
        )

    base = baseline_row.iloc[0]
    out["delta_mean_cluster_dist_vs_baseline"] = out["mean_cluster_dist"] - base["mean_cluster_dist"]
    out["delta_mixed_rate_vs_baseline"] = out["mixed_rate"] - base["mixed_rate"]
    out["delta_hhi_vs_baseline"] = out["hhi_size_concentration"] - base["hhi_size_concentration"]
    out["delta_largest_cluster_ratio_vs_baseline"] = out["largest_cluster_ratio"] - base["largest_cluster_ratio"]

    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and rank ObjectFusion candidate runs from even-split scorecard exports."
    )
    parser.add_argument(
        "--metrics-csv",
        required=True,
        help="CSV exported from section 10 of analysis/objectfusion_scorecard_queries.sql",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/objectfusion_candidate_ranking.csv",
        help="Path for ranked output CSV",
    )
    parser.add_argument(
        "--baseline",
        default="baseline_even",
        help="candidate_name to use as baseline for delta columns",
    )
    parser.add_argument(
        "--visual-csv",
        default=None,
        help="Optional CSV with candidate_name,visual_score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics_csv)
    output_path = Path(args.output_csv)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)

    visual = None
    if args.visual_csv:
        visual_path = Path(args.visual_csv)
        if not visual_path.exists():
            raise FileNotFoundError(f"Visual CSV not found: {visual_path}")
        visual = load_visual_scores(visual_path)

    ranked = build_scorecard(metrics, baseline_name=args.baseline, visual_df=visual)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)

    display_cols = [
        "rank",
        "candidate_name",
        "composite_score",
        "mean_cluster_dist",
        "mixed_rate",
        "hhi_size_concentration",
        "largest_cluster_ratio",
        "delta_mean_cluster_dist_vs_baseline",
        "delta_mixed_rate_vs_baseline",
    ]
    print(ranked[display_cols].to_string(index=False))
    print(f"\nSaved ranking CSV: {output_path}")


if __name__ == "__main__":
    main()
