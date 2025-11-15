#!/usr/bin/env python3
"""
Scatter plot of (X_metric, total_tests_run) for each
batching×bisection combo, with a Pareto frontier overlay.

Now supports choosing the horizontal axis:
  - mean feedback time (hr)
  - mean time to culprit (hr)
  - max time to culprit (hr)

Notes:
- Points are color/marker-coded by strategy name; labels are in the legend.
- "Exhaustive Testing (ET)" is excluded from the plot (and Pareto).
- Pareto frontier assumes we minimize both axes.

Usage:
  python risk_strats_scatter_pareto.py --json_path path/to/results.json \
      [--xaxis mean_ttc|mft|max_ttc] [--out fig.png]

Defaults:
  --xaxis mean_ttc
"""

import argparse
import json
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools

EXCLUDE_NAMES = {"Exhaustive Testing (ET)"}

# Map CLI choice -> (json_key, pretty_label)
X_METRIC_MAP = {
    "mft": ("mean_feedback_time_hr", "Mean feedback time (hr) (lower is better)"),
    "mean_ttc": ("mean_time_to_culprit_hr", "Mean time to culprit (hr) (lower is better)"),
    "max_ttc": ("max_time_to_culprit_hr", "Max time to culprit (hr) (lower is better)"),
}


def load_points(path: str, x_key: str) -> Tuple[List[Tuple[float, float, str]], dict]:
    """Return (points, meta) where points is [(x, y, name), ...].
    Uses chosen x_key for X and 'total_tests_run' for Y.
    Only entries with both fields present are used.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for name, val in data.items():
        if name in EXCLUDE_NAMES:
            continue
        if not isinstance(val, dict):
            continue
        if x_key in val and "total_tests_run" in val:
            try:
                x = float(val[x_key])                 # chosen time metric on X (minimize)
                y = float(val["total_tests_run"])     # total tests on Y (minimize)
                points.append((x, y, name))
            except (TypeError, ValueError):
                pass
    return points, data


def pareto_front(points: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """Compute Pareto frontier minimizing both X and Y.
    Returns points in ascending X order.
    """
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    front = []
    best_y = float("inf")
    for x, y, name in pts:
        if y <= best_y:
            front.append((x, y, name))
            best_y = y
    return front


def main():
    ap = argparse.ArgumentParser(description="Plot strategy scatter with Pareto frontier.")
    ap.add_argument("--json_path", required=True, help="Path to results JSON file")
    ap.add_argument(
        "--xaxis",
        choices=list(X_METRIC_MAP.keys()),
        default="mean_ttc",
        help="Horizontal axis metric: mft (mean feedback time), mean_ttc, or max_ttc",
    )
    ap.add_argument("--out", help="Optional output image path (e.g., plot.png)")
    args = ap.parse_args()

    x_key, x_label = X_METRIC_MAP[args.xaxis]

    points, meta = load_points(args.json_path, x_key)
    if not points:
        print("No plottable points found in JSON for the chosen x-axis metric.", file=sys.stderr)
        sys.exit(1)

    name_to_xy = {name: (x, y) for x, y, name in points}

    names = sorted({name for *_rest, name in points})
    cmap = plt.get_cmap("tab20", len(names))
    marker_cycle = itertools.cycle(['o', 's', '^', 'v', '<', '>', 'D', 'd', 'X', 'P', '+', '1', '2', '*'])
    name_to_style = {name: {"color": cmap(i), "marker": next(marker_cycle)} for i, name in enumerate(names)}

    plt.figure(figsize=(10, 8))

    for x, y, name in points:
        style = name_to_style[name]
        plt.scatter(x, y, s=150, zorder=2, color=style["color"], marker=style["marker"])

    # Pareto frontier
    front = pareto_front(points)
    if len(front) >= 2:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        plt.plot(fx, fy, linewidth=1.6, zorder=3)
        # small tag near the first point to indicate the line meaning (kept subtle)
        plt.annotate("Pareto", (fx[0], fy[0]), textcoords="offset points", xytext=(6, -10), fontsize=9)
        # Label each frontier point with its combo name (small, offset to reduce overlap)
        for i, (px, py, pname) in enumerate(front):
            va = "bottom" if (i % 2 == 0) else "top"
            plt.annotate(
                pname,
                (px, py),
                textcoords="offset points",
                xytext=(6, 6 if va == "bottom" else -8),
                ha="left",
                va=va,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
            )

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel("Total tests run (lower is better)", fontsize=16)
    # plt.title(f"Batching × Bisection Strategies: {x_label.split(' (')[0]} vs Total Tests", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle=":", linewidth=0.6)

    proxies = [
        Line2D(
            [0], [0],
            linestyle='none',
            marker=name_to_style[n]["marker"],
            markerfacecolor=name_to_style[n]["color"],
            markeredgecolor=name_to_style[n]["color"],
            markersize=8,
            label=n
        )
        for n in names
    ]
    plt.legend(
        handles=proxies,
        title="Strategy (color + shape)",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8
    )

    # Make space on the right for the legend
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
