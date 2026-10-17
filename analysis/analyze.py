"""Reproduce the Go2 comparison from the original TensorBoard files.

Run from the submission root: python analysis/analyze.py
Only safe YAML values are decoded; Python object tags are never executed.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "baseline": "2026-10-13_09-28-07",
    "wide_300": "2026-10-13_10-21-07_wide_300",
    "wide_900": "2026-10-14_10-50-41_wide_900",
}
TAGS = {
    "error_xy": "Metrics/base_velocity/error_vel_xy",
    "success": "Metrics/success_rate",
    "base_contact": "Episode_Termination/base_contact",
    "reward": "Train/mean_reward",
    "error_yaw": "Metrics/base_velocity/error_vel_yaw",
    "time_out": "Episode_Termination/time_out",
}

class ValueLoader(yaml.SafeLoader):
    """Decode Isaac Lab's Python-tagged configuration as plain values."""

def plain_python_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)

ValueLoader.add_multi_constructor("tag:yaml.org,2002:python/", plain_python_tag)

def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            result += differences(a.get(key), b.get(key), f"{path}.{key}".strip("."))
        return result
    return [] if a == b else [{"path": path, "before": a, "after": b}]

def rolling_mean(values, window=20):
    # Trailing average with a shorter available window at the beginning.
    return [statistics.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]

def main():
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    summary = {}
    configs = {}
    all_rows = []
    curves = {}
    hashes = {}
    for label, directory in RUNS.items():
        path = ROOT / "runs" / directory
        env = yaml.load((path / "params/env.yaml").read_text(), Loader=ValueLoader)
        agent = yaml.safe_load((path / "params/agent.yaml").read_text())
        configs[label] = {"env": env, "agent": agent}
        events = sorted(path.glob("events.out.tfevents*"))
        if len(events) != 1:
            raise ValueError(f"Expected one event file for {label}, got {len(events)}")
        accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
        accumulator.Reload()
        available = accumulator.Tags()["scalars"]
        n = agent["max_iterations"]
        metrics = {}
        for key, tag in TAGS.items():
            if tag not in available:
                raise ValueError(f"Missing {tag} in {label}; available: {available}")
            points = accumulator.Scalars(tag)
            by_step = {p.step: p for p in points}
            if len(points) != n or sorted(by_step) != list(range(n)):
                raise ValueError(f"Incomplete/duplicate {tag} in {label}")
            values = [by_step[i].value for i in range(n)]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values for {label}/{tag}")
            if key in {"success", "base_contact", "time_out"} and not all(0 <= x <= 1 for x in values):
                raise ValueError(f"Rate outside [0,1] in {label}/{tag}")
            metrics[key] = values
            for step in range(n):
                p = by_step[step]
                all_rows.append({"run": label, "tag": tag, "step": step,
                                 "value": p.value, "wall_time": p.wall_time})
        curves[label] = metrics
        final = {k: v[-1] for k, v in metrics.items()}
        late = {k: statistics.mean(v[-50:]) for k, v in metrics.items()}
        summary[label] = {
            "directory": directory, "iterations": n, "seed": agent["seed"],
            "num_envs": env["scene"]["num_envs"],
            "steps_per_iteration": env["scene"]["num_envs"] * agent["num_steps_per_env"],
            "total_steps": n * env["scene"]["num_envs"] * agent["num_steps_per_env"],
            "ranges": env["commands"]["base_velocity"]["ranges"],
            "final": final, "last_50_mean": late,
            "last_50_steps": [n-50, n-1],
            "all_scalar_tags": available,
        }
        for f in sorted(path.rglob("*")):
            if f.is_file(): hashes[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    config_diffs = {}
    for a, b in [("baseline", "wide_300"), ("wide_300", "wide_900")]:
        diff = differences(configs[a], configs[b])
        config_diffs[f"{a}_to_{b}"] = diff
        allowed = {"agent.run_name", "env.log_dir", "env.commands.base_velocity.ranges.lin_vel_x",
                   "env.commands.base_velocity.ranges.lin_vel_y"} if a == "baseline" else {"agent.run_name", "env.log_dir", "agent.max_iterations"}
        unexpected = [x for x in diff if x["path"] not in allowed]
        if unexpected: raise ValueError(f"Unexpected config differences: {unexpected}")
    summary["method"] = {
        "late_window": "Unweighted mean of the last 50 scalar log points per run.",
        "chart_smoothing": "Raw observations plus a trailing 20-iteration mean; no extrapolation.",
        "limits": "Training logs, one seed per condition; not pooled episode rates or held-out evaluation."
    }
    (ROOT / "analysis/summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (ROOT / "analysis/config_differences.json").write_text(json.dumps(config_diffs, indent=2)+"\n")
    (ROOT / "analysis/source_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
    with (ROOT / "analysis/scalars.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "tag", "step", "value", "wall_time"])
        w.writeheader(); w.writerows(all_rows)
    with (ROOT / "analysis/late_window_summary.csv").open("w", newline="") as f:
        fields = ["run", "start_step", "end_step"] + list(TAGS)
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for label in RUNS:
            n = summary[label]["iterations"]
            w.writerow({"run": label, "start_step": n-50, "end_step": n-1,
                        **summary[label]["last_50_mean"]})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#334155", "text.color": "#0f172a",
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    colors = {"baseline": "#2563eb", "wide_300": "#e87915", "wide_900": "#0f9d83"}
    labels = {"baseline": "Baseline · 300", "wide_300": "Wide · 300", "wide_900": "Wide · 900"}
    specs = [("error_xy", "Movement speed error", "m/s · lower is better", 1.0),
             ("success", "Success rate", "% · higher is better", 100.0),
             ("base_contact", "Body-contact termination rate", "% · lower is better", 100.0)]
    for x_limit, filename in [(900, "learning_curves.png"), (300, "first_300_comparison.png")]:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True)
        for ax, (key, title, unit, multiplier) in zip(axes, specs):
            for label in RUNS:
                values = [x * multiplier for x in curves[label][key]]
                count = min(len(values), x_limit)
                x = np.arange(count)
                ax.plot(x, values[:count], color=colors[label], alpha=.16, linewidth=.7)
                ax.plot(x, rolling_mean(values)[:count], color=colors[label], linewidth=2, label=labels[label])
            ax.set_title(title, loc="left", pad=9)
            ax.set_ylabel(unit)
            ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
            ax.set_xlim(0, x_limit-1)
            ax.set_ylim(bottom=0)
            if key == "success": ax.set_ylim(0, 105)
            if x_limit == 900: ax.axvline(299, color="#64748b", linestyle="--", linewidth=.8)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower left", bbox_to_anchor=(.075, .875), ncol=3, frameon=False)
        axes[-1].set_xlabel("Training iteration (zero-based)")
        heading = "Wider speed commands remain difficult after 900 iterations" if x_limit == 900 else "The baseline learns speed tracking within 300 iterations"
        fig.suptitle(heading, x=.08, ha="left", fontsize=16, fontweight="bold")
        fig.text(.08, .925, "Faint lines: raw training logs. Solid lines: trailing 20-iteration means. Seed 42 for each run.", fontsize=10, color="#475569")
        fig.text(.08, .025, "300-iteration runs end at 299; they are not extrapolated. Rates are training-log metrics, not held-out evaluation.", fontsize=9, color="#475569")
        fig.subplots_adjust(left=.12, right=.97, top=.835, bottom=.085, hspace=.38)
        fig.savefig(ROOT / "figures" / filename, dpi=180)
        plt.close(fig)
    print(json.dumps({k: {"final":summary[k]["final"], "last_50_mean":summary[k]["last_50_mean"]} for k in RUNS}, indent=2))
    print("Verified config differences:", json.dumps(config_diffs, indent=2))

if __name__ == "__main__":
    main()
