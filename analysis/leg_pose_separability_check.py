#!/usr/bin/env python3
"""Offline validation for the leg-pose separability heuristic (tools_clustering.py).

Pulls leg-shape features (LocationHandsFeet) for one cluster and reports whether
the gap-based separability check would split it, without touching make_video.py.

Usage:
    python analysis/leg_pose_separability_check.py --cluster-id 64
    python analysis/leg_pose_separability_check.py --cluster-id 64 --cluster-table ArmsPoses3D  --min-gap-ratio 0.3
"""
import argparse
import os
import sys

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from mp_db_io import DataIO
from tools_clustering import ToolsClustering


def fetch_cluster_leg_features(engine, cluster_table, cluster_id, helper_table):
    junction_table = f"Images{cluster_table}"
    sql = text(f"""
        SELECT j.image_id, lhf.leg_extension_max, lhf.leg_extension_min,
               lhf.leg_asymmetry, lhf.visible_leg_count
        FROM {junction_table} j
        LEFT JOIN LocationHandsFeet lhf ON j.image_id = lhf.image_id
        LEFT JOIN {helper_table} h ON j.image_id = h.image_id
        WHERE j.cluster_id = :cluster_id
    """)
    # print sql statement for debugging
    print(f"Executing SQL:\n{sql}\nwith cluster_id={cluster_id}")
    with engine.connect() as conn:
        rows = conn.execute(sql, {"cluster_id": cluster_id}).mappings().all()
    return pd.DataFrame(rows)


def plot_leg_distribution(df, boundary=None, output_path=None, title=None):
    visible_mask = pd.to_numeric(df["visible_leg_count"], errors="coerce").fillna(0) > 0
    values = pd.to_numeric(df.loc[visible_mask, "leg_extension_max"], errors="coerce").dropna()
    if values.empty:
        print("No visible leg_extension_max values to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = min(60, max(20, len(values) // 20))
    ax.hist(values, bins=bins, color="#5b8ff9", edgecolor="black", alpha=0.8)

    if boundary is not None:
        ax.axvline(
            boundary,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"boundary={boundary:.3f}",
        )
        ax.legend()

    ax.set_title(title or "Distribution of leg_extension_max for visible legs")
    ax.set_xlabel("leg_extension_max")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=180)
        print(f"Saved histogram to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-id", type=int, required=True, help="cluster_id in Images<cluster-table>")
    parser.add_argument("--cluster-table", default="ArmsPoses3D", help="cluster family, e.g. ArmsPoses3D")
    parser.add_argument("--helper-table", default="SegmentHelper_TheGym", help="segment helper to narrow query")
    parser.add_argument("--floor-pct", type=float, default=5.0)
    parser.add_argument("--min-bucket-size", type=int, default=20)
    parser.add_argument("--min-gap-ratio", type=float, default=0.35)
    parser.add_argument("--plot", action="store_true", help="show/save a histogram of leg_extension_max")
    parser.add_argument("--plot-path", default="leg_pose_distribution.png", help="output PNG path when --plot is used")
    args = parser.parse_args()

    io = DataIO()
    db = io.db
    engine = create_engine(
        f"mysql+pymysql://{db['user']}:{db['pass']}@/{db['name']}?unix_socket={db['unix_socket']}",
        pool_pre_ping=True,
        pool_recycle=600,
        poolclass=NullPool,
    )

    df = fetch_cluster_leg_features(engine, args.cluster_table, args.cluster_id, args.helper_table)
    print(f"Fetched {len(df)} rows for {args.cluster_table} cluster_id={args.cluster_id}")

    cl = ToolsClustering(args.cluster_table, VERBOSE=True)
    result = cl.assess_leg_pose_separability(
        df,
        floor_pct=args.floor_pct,
        min_bucket_size=args.min_bucket_size,
        min_gap_ratio=args.min_gap_ratio,
        cluster_label=f"{args.cluster_table}:{args.cluster_id}",
    )

    print("\n=== Separability result ===")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if args.plot:
        plot_leg_distribution(
            df,
            boundary=result.get("boundary"),
            output_path=args.plot_path,
            title=f"{args.cluster_table}:{args.cluster_id} leg_extension_max distribution",
        )

    if result["is_separable"]:
        labels = cl.label_by_leg_pose(df, result["boundary"])
        print("\n=== Label breakdown ===")
        print(labels.value_counts())
    else:
        print("\nNot separable under current thresholds; no split recommended.")

    engine.dispose()


if __name__ == "__main__":
    main()
