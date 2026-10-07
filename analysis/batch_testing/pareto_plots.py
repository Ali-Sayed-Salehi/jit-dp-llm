"""Optional, headless plots of sampled tests-versus-max-TTC trade-offs."""

import csv
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import re


logger = logging.getLogger(__name__)
X_KEY = "total_tests_run"
Y_KEY = "max_time_to_culprit_hr"
BASELINE = "TWSB + PAR"
ET = "Exhaustive Testing (ET)"


def require_plotting_dependency():
    """Check before expensive simulations without importing plotting libraries."""
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError(
            "--plot-pareto-fronts requires matplotlib. Install it in the Python "
            "environment used to run simulation.py (python -m pip install matplotlib)."
        )


def trial_data_path(results_path):
    path = Path(results_path)
    return path.with_name(path.stem + "_pareto_trials.json")


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_trial_data(results_path, trials):
    """Keep trial data separate from the existing EVAL results schema."""
    path = Path(results_path)
    payload = {
        "schema_version": 1,
        "split": "EVAL",
        "x_metric": X_KEY,
        "y_metric": Y_KEY,
        "eval_results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "trials": trials,
    }
    destination = trial_data_path(path)
    destination.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved EVAL trial data to %s", destination)
    return destination


def load_trial_data(results_path):
    """Reuse recorded trials only when they belong to the saved EVAL file."""
    path = Path(results_path)
    destination = trial_data_path(path)
    reason = None
    if not destination.exists():
        reason = "saved EVAL trial data is missing"
    else:
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                payload.get("schema_version") != 1
                or payload.get("eval_results_sha256") != expected
                or not isinstance(payload.get("trials"), dict)
            ):
                reason = "saved EVAL trial data does not match this EVAL results file"
            else:
                return payload["trials"]
        except (OSError, ValueError, AttributeError) as exc:
            reason = f"saved EVAL trial data could not be read: {exc}"
    logger.warning(
        "%s; EVAL tuning fronts cannot be reconstructed. Plotting the selected "
        "configurations instead. Run EVAL with --plot-pareto-fronts to retain trials.",
        reason,
    )
    return None


def is_plottable(point):
    if point.get("found_all_regressors") is not True:
        return False
    try:
        return all(
            math.isfinite(float(point[key])) and float(point[key]) >= 0
            for key in (X_KEY, Y_KEY)
        )
    except (KeyError, TypeError, ValueError):
        return False


def pareto_indices(points):
    """Minimize both axes; identical points tie, equal-Y higher-X points lose."""
    ordered = sorted(
        (i for i, point in enumerate(points) if is_plottable(point)),
        key=lambda i: (float(points[i][X_KEY]), float(points[i][Y_KEY])),
    )
    front = set()
    best_y = math.inf
    best_xy = None
    for i in ordered:
        xy = (float(points[i][X_KEY]), float(points[i][Y_KEY]))
        if xy[1] < best_y:
            best_y, best_xy = xy[1], xy
            front.add(i)
        elif xy == best_xy:
            front.add(i)
    return front


def _included(name, selected_batching, selected_bisection, include_et):
    if name == ET:
        return include_et
    if " + " not in name:
        return False
    batching, bisection = name.split(" + ", 1)
    return (
        (selected_batching is None or batching in selected_batching)
        and (selected_bisection is None or bisection in selected_bisection)
    )


def collect_points(results, trials=None, *, selected_batching=None,
                   selected_bisection=None, include_et=True):
    """Use all recorded trials, even if tuning minimized a different metric."""
    selected = {}
    for name, entry in results.items():
        name = BASELINE if name == "Baseline (TWSB + PAR)" else name
        if not isinstance(entry, dict) or not _included(
            name, selected_batching, selected_bisection, include_et
        ):
            continue
        if is_plottable(entry):
            selected[name] = {
                "strategy": name, "trial_number": None,
                "params": entry.get("best_params", entry.get("best_params_from_eval", {})),
                X_KEY: entry[X_KEY], Y_KEY: entry[Y_KEY],
                "found_all_regressors": True, "selected": True,
                "kind": "reference" if name.startswith("TWSB + ") or name == ET else "selected",
            }

    points = []
    for name, records in (trials or {}).items():
        if not _included(name, selected_batching, selected_bisection, include_et):
            continue
        choice = selected.get(name)
        for record in records:
            if not is_plottable(record):
                continue
            point = {**record, "strategy": name, "kind": "trial", "selected": False}
            if choice is not None:
                point["selected"] = all(
                    point.get(key) == choice.get(key) for key in ("params", X_KEY, Y_KEY)
                )
            points.append(point)

    # Add fixed reference points and selected replays not already represented.
    for name, choice in selected.items():
        if not any(p["strategy"] == name and p["selected"] for p in points):
            points.append(choice)
    return points


def _write_points_csv(path, points):
    overall = pareto_indices(points)
    within = set()
    for name in {p["strategy"] for p in points}:
        indices = [i for i, p in enumerate(points) if p["strategy"] == name]
        within.update(indices[i] for i in pareto_indices([points[j] for j in indices]))
    columns = ["strategy", "trial_number", "kind", X_KEY, Y_KEY,
               "selected", "pareto_within_strategy", "pareto_overall", "parameters_json"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, point in enumerate(points):
            writer.writerow({
                **{key: point.get(key) for key in columns[:6]},
                "pareto_within_strategy": i in within,
                "pareto_overall": i in overall,
                "parameters_json": json.dumps(point.get("params", {}), sort_keys=True),
            })


def _format_test_count(value, _position=None):
    """Keep test-count ticks compact on both linear and logarithmic axes."""
    for scale, suffix in ((1_000_000, "M"), (1_000, "k")):
        if abs(value) >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _strategy_label(name, names):
    """Omit the bisection suffix only when the run shows one method."""
    methods = {candidate.split(" + ", 1)[1] for candidate in names if " + " in candidate}
    return name.split(" + ", 1)[0] if len(methods) == 1 else name


def _render(points, destination, title, *, tuning_front=False, style_names=None):
    # Import only after all simulations complete; no display server is needed.
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib.ticker import FuncFormatter, NullFormatter

    names = sorted({point["strategy"] for point in points})
    style_names = style_names or names
    fig = Figure(figsize=(8, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    handles = []
    markers = ("o", "s", "^", "D", "v", "P", "h", "<", ">", "p", "8", "d",
               "X", "H", "+", "x", "1", "2", "3", "4")
    for name in names:
        n = style_names.index(name)
        color = colormaps["tab20"]((2 * (n % 10) + (n % 20) // 10))
        marker = markers[(n // 20 if tuning_front else n) % len(markers)]
        group = [point for point in points if point["strategy"] == name]
        label = _strategy_label(name, style_names)
        if tuning_front:
            ax.scatter([p[X_KEY] for p in group], [p[Y_KEY] for p in group],
                       color=color, marker=marker, s=24, alpha=0.35, zorder=2)
            front = [group[i] for i in pareto_indices(group)]
            front = sorted({(p[X_KEY], p[Y_KEY]) for p in front})
            if front:
                ax.plot([xy[0] for xy in front], [xy[1] for xy in front],
                        color=color, marker=marker, markersize=4, linewidth=1.2, zorder=3)
            handles.append(Line2D([0], [0], color=color, marker=marker, label=label))
        else:
            # Hollow, varied shapes reveal coincident configurations without
            # moving their coordinates or hiding earlier points under a fill.
            colors = ({"facecolors": "none", "edgecolors": color}
                      if MarkerStyle(marker).is_filled() else {"color": color})
            ax.scatter([p[X_KEY] for p in group], [p[Y_KEY] for p in group],
                       marker=marker, s=110, linewidths=1.7, zorder=5, **colors)
            handles.append(Line2D([0], [0], color=color, marker=marker,
                                  linestyle="none", markerfacecolor="none",
                                  markersize=9, markeredgewidth=1.7, label=label))

    overall = sorted({(points[i][X_KEY], points[i][Y_KEY]) for i in pareto_indices(points)})
    if len(overall) >= 2:
        ax.plot([xy[0] for xy in overall], [xy[1] for xy in overall],
                color="black", linestyle="--", linewidth=1.4, zorder=4)
    if tuning_front:
        ax.scatter([xy[0] for xy in overall], [xy[1] for xy in overall],
                   facecolors="none", edgecolors="black", s=75, linewidths=1.1, zorder=4)
    if tuning_front or len(overall) >= 2:
        handles.append(Line2D([0], [0], color="black", linestyle="--",
                              marker="o" if tuning_front else None, markerfacecolor="none",
                              label="Sampled Pareto front" if tuning_front else "Pareto front of shown configurations"))
    for axis, key in (("x", X_KEY), ("y", Y_KEY)):
        values = [float(p[key]) for p in points]
        if min(values) > 0 and max(values) / min(values) >= 100:
            getattr(ax, f"set_{axis}scale")("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_format_test_count))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Total tests" + (" (log scale)" if ax.get_xscale() == "log" else ""),
                  fontsize=12)
    ax.set_ylabel("Maximum TTC (hours)" + (" (log scale)" if ax.get_yscale() == "log" else ""),
                  fontsize=12)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(handles=handles, loc="upper right", fontsize=10,
              ncols=max(1, math.ceil(len(handles) / 20)))
    fig.savefig(str(destination) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(destination) + ".pdf", bbox_inches="tight")
    fig.clear()


def write_pareto_plots(results_path, results, *, split, trials=None,
                       selected_batching=None, selected_bisection=None, include_et=True):
    """Write images and a CSV beside the associated results JSON."""
    path = Path(results_path)
    points = collect_points(results, trials, selected_batching=selected_batching,
                            selected_bisection=selected_bisection, include_et=include_et)
    # Exclude both baselines from plots, axis ranges, and the plotted-point CSV.
    points = [point for point in points
              if point["strategy"] != ET and not point["strategy"].startswith("TWSB + ")]
    csv_path = path.with_name(path.stem + "_pareto_points.csv")
    _write_points_csv(csv_path, points)
    if not points:
        logger.warning("No feasible finite configurations to plot on %s", split)
        return []

    tuning_front = split == "EVAL" and trials is not None
    suffix = "_pareto_all" if tuning_front else "_pareto_selected"
    destination = path.with_name(path.stem + suffix)
    title = f"{split}: sampled trade-offs" if tuning_front else f"{split}: strategy comparison"
    style_names = sorted({point["strategy"] for point in points})
    _render(points, destination, title, tuning_front=tuning_front, style_names=style_names)
    written = [destination]
    if tuning_front:
        for name in sorted({point["strategy"] for point in points}):
            group = [point for point in points if point["strategy"] == name]
            slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
            destination = path.with_name(path.stem + "_pareto_" + slug)
            _render(group, destination, f"EVAL: {_strategy_label(name, style_names)}", tuning_front=True,
                    style_names=style_names)
            written.append(destination)
    logger.info("Saved %s Pareto plots and point data beside %s", split, path)
    return written
