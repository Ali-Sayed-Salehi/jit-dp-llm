#!/usr/bin/env python3
"""
Scatter plot of (total_tests_run, max_time_to_culprit_hr) for each
batching×bisection combo, with a Pareto frontier overlay.

Changes from previous version:
- Labels moved outside the axes: we use a color-coded legend.
- Each combo gets its own color; points are not annotated on the plot.
- "Exhaustive Testing (ET)" is excluded from the plot (and Pareto).

Usage:
  python risk_strats_scatter_pareto.py path/to/results.json [--out fig.png]

The JSON is expected to be a dict mapping combo names to metrics.
Non-metric keys (e.g., "final_window", "best_by_*") are ignored.
"""
import argparse
import json
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools


EXCLUDE_NAMES = {"Exhaustive Testing (ET)"}


def load_points(path: str) -> Tuple[List[Tuple[float, float, str]], dict]:
    """Return (points, meta) where points is [(x, y, name), ...].
    Only entries with both 'total_tests_run' and 'max_time_to_culprit_hr' are used.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for name, val in data.items():
        if name in EXCLUDE_NAMES:
            continue
        if not isinstance(val, dict):
            continue
        if "total_tests_run" in val and "max_time_to_culprit_hr" in val:
            try:
                x = float(val["total_tests_run"])  # tests on X (minimize)
                y = float(val["max_time_to_culprit_hr"])  # max TTC on Y (minimize)
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
    ap.add_argument("--json_path", help="Path to results JSON file")
    ap.add_argument("--out", help="Optional output image path (e.g., plot.png)")
    args = ap.parse_args()

    points, meta = load_points(args.json_path)
    if not points:
        print("No plottable points found in JSON.", file=sys.stderr)
        sys.exit(1)

    # Identify 'best' from metadata if present
    best_total = meta.get("best_by_total_tests")
    best_max = meta.get("best_by_max_ttc")

    # Build a quick index from name -> (x,y)
    name_to_xy = {name: (x, y) for x, y, name in points}

    # Unique, stable order of names for color assignment
    names = sorted({name for *_rest, name in points})
    # Use pyplot.get_cmap to avoid the Matplotlib 3.7 deprecation warning
    cmap = plt.get_cmap("tab20", len(names))
    # Avoid using '*' because we reserve it for the "best" highlight
    marker_cycle = itertools.cycle(['o', 's', '^', 'v', '<', '>', 'D', 'd', 'X', 'P', '+', '1', '2', '*'])
    name_to_style = {name: {"color": cmap(i), "marker": next(marker_cycle)} for i, name in enumerate(names)}

    # Figure and axes
    plt.figure(figsize=(22, 6))

    # Plot all points with colors; use legend for labels (outside the axes)
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

    plt.xlabel("Total tests run (lower is better)", fontsize=16)
    plt.ylabel("Max time to culprit (hr) (lower is better)", fontsize=16)
    plt.title("Batching × Bisection Strategies: Tests vs Max TTC", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, linestyle=":", linewidth=0.6)

    # Legend with both color and marker, placed outside on the right
    proxies = [
        Line2D([0], [0], linestyle='none', marker=name_to_style[n]["marker"],
               markerfacecolor=name_to_style[n]["color"], markeredgecolor=name_to_style[n]["color"], markersize=8,
               label=n)
        for n in names
    ]
    leg = plt.legend(handles=proxies, title="Strategy (color + shape)",
                     bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=8)

    # Make space on the right for the legend
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    if args.out:
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
