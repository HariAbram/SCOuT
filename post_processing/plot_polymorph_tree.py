#!/usr/bin/env python3
"""Draw per-benchmark polyMorph search trees from trial CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polymorph_results import config_dirs, first_existing, opt_and_benchmark, safe_float, slug, transform_label


JsonDict = dict[str, Any]


@dataclass
class PrefixNode:
    key: str
    parent: str | None
    depth: int
    last_transform: str
    visits: int = 0
    terminal_visits: int = 0
    pruned_visits: int = 0
    speedups: list[float] = field(default_factory=list)
    objectives: list[float] = field(default_factory=list)

    @property
    def best_speedup(self) -> float | None:
        return max(self.speedups) if self.speedups else None

    @property
    def mean_speedup(self) -> float | None:
        return sum(self.speedups) / len(self.speedups) if self.speedups else None

    @property
    def best_objective(self) -> float | None:
        return min(self.objectives) if self.objectives else None


def require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/scout-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/scout-cache")
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for tree figures. Install it with "
            "`pip install matplotlib`."
        ) from exc
    return plt, cm, colors


def compact_transform(transform: JsonDict) -> JsonDict:
    return {
        "scop": transform.get("scop"),
        "node": transform.get("node"),
        "tr": transform.get("tr") or transform.get("transform"),
        "args": transform.get("args", []),
    }


def prefix_key(transforms: list[JsonDict]) -> str:
    compact = [compact_transform(item) for item in transforms]
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def parse_transforms(value: str | None) -> list[JsonDict]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def load_prefix_tree(trials_path: Path) -> dict[str, PrefixNode]:
    root_key = "root"
    nodes: dict[str, PrefixNode] = {
        root_key: PrefixNode(root_key, None, 0, "baseline")
    }
    with trials_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            state = str(row.get("state") or "").upper()
            transforms = parse_transforms(row.get("transforms"))
            speedup = safe_float(row.get("speedup"))
            objective = safe_float(row.get("objective"))
            if not transforms:
                continue
            for depth in range(1, len(transforms) + 1):
                prefix = transforms[:depth]
                key = prefix_key(prefix)
                parent = root_key if depth == 1 else prefix_key(transforms[: depth - 1])
                node = nodes.setdefault(
                    key,
                    PrefixNode(
                        key=key,
                        parent=parent,
                        depth=depth,
                        last_transform=transform_label(prefix[-1]),
                    ),
                )
                node.visits += 1
                if state == "PRUNED":
                    node.pruned_visits += 1
                if depth == len(transforms):
                    node.terminal_visits += 1
                if speedup is not None:
                    node.speedups.append(speedup)
                if objective is not None:
                    node.objectives.append(objective)
    return nodes


def rank_nodes(nodes: dict[str, PrefixNode], metric: str) -> dict[str, int]:
    scored: list[tuple[float, str]] = []
    for key, node in nodes.items():
        if key == "root":
            continue
        value = node.best_speedup if metric == "best_speedup" else node.mean_speedup
        if value is not None:
            scored.append((value, key))
    scored.sort(reverse=True)
    return {key: rank + 1 for rank, (_, key) in enumerate(scored)}


def visible_nodes(
    nodes: dict[str, PrefixNode],
    ranks: dict[str, int],
    *,
    max_prefixes: int,
    max_depth: int,
) -> set[str]:
    root_key = "root"
    visible = {root_key}
    candidates = [
        key for key in ranks
        if max_depth <= 0 or nodes[key].depth <= max_depth
    ]
    if max_prefixes > 0:
        candidates = sorted(candidates, key=lambda key: ranks[key])[:max_prefixes]
    for key in candidates:
        current: str | None = key
        while current:
            visible.add(current)
            current = nodes[current].parent
    for key, node in nodes.items():
        if max_depth > 0 and node.depth > max_depth:
            continue
        if node.parent in visible and (max_prefixes <= 0 or len(visible) < max_prefixes * 2):
            visible.add(key)
    return visible


def node_metric(node: PrefixNode, metric: str) -> float | None:
    if node.key == "root":
        return 1.0
    return node.best_speedup if metric == "best_speedup" else node.mean_speedup


def shorten_label(text: str, width: int = 30) -> str:
    if len(text) <= width:
        return text
    return f"{text[: width - 3]}..."


def prefix_transform_labels(key: str) -> list[str]:
    if key == "root":
        return ["baseline"]
    try:
        transforms = json.loads(key)
    except json.JSONDecodeError:
        return [key]
    if not isinstance(transforms, list):
        return [key]
    return [
        transform_label(item)
        for item in transforms
        if isinstance(item, dict)
    ]


def prefix_table_text(
    nodes: dict[str, PrefixNode],
    ranks: dict[str, int],
    *,
    metric: str,
    limit: int,
) -> str:
    ranked_keys = sorted(ranks, key=lambda key: ranks[key])
    if limit > 0:
        ranked_keys = ranked_keys[:limit]
    lines = ["Rank  Depth  Speedup  Prefix"]
    lines.append("----  -----  -------  ------")
    for key in ranked_keys:
        node = nodes[key]
        value = node_metric(node, metric)
        value_text = "-" if value is None else f"{value:.3f}x"
        prefix = " -> ".join(prefix_transform_labels(key))
        lines.append(
            f"{ranks[key]:>4}  {node.depth:>5}  {value_text:>7}  {shorten_label(prefix, 74)}"
        )
    return "\n".join(lines)


def best_path_keys(nodes: dict[str, PrefixNode], ranks: dict[str, int]) -> set[str]:
    if not ranks:
        return set()
    best = min(ranks, key=lambda key: ranks[key])
    path: set[str] = set()
    current: str | None = best
    while current:
        path.add(current)
        current = nodes[current].parent
    return path


def node_text(node: PrefixNode, ranks: dict[str, int], metric: str) -> str:
    value = node_metric(node, metric)
    value_text = "-" if value is None else f"{value:.2f}x"
    if node.key == "root":
        return "root\n1.00x"
    return f"#{ranks.get(node.key, '?')}\n{value_text}"


def inline_node_text(node: PrefixNode, ranks: dict[str, int], metric: str) -> str:
    value = node_metric(node, metric)
    value_text = "-" if value is None else f"{value:.3f}x"
    if node.key == "root":
        return "root baseline"
    return f"#{ranks.get(node.key, '?')} {value_text} {shorten_label(node.last_transform, 26)}"


def draw_tree(
    plt,
    cm,
    colors,
    nodes: dict[str, PrefixNode],
    out_path: Path,
    *,
    title: str,
    metric: str,
    max_prefixes: int,
    max_depth: int,
    label_top: int,
    label_mode: str,
    dpi: int,
) -> bool:
    ranks = rank_nodes(nodes, metric)
    if not ranks:
        return False
    visible = visible_nodes(nodes, ranks, max_prefixes=max_prefixes, max_depth=max_depth)
    visible_nodes_list = [nodes[key] for key in visible]
    depths = sorted({node.depth for node in visible_nodes_list})
    by_depth: dict[int, list[PrefixNode]] = {}
    for depth in depths:
        level = [node for node in visible_nodes_list if node.depth == depth]
        level.sort(
            key=lambda node: (
                ranks.get(node.key, math.inf),
                node.last_transform,
            )
        )
        by_depth[depth] = level

    coords: dict[str, tuple[float, float]] = {}
    max_level = max(len(level) for level in by_depth.values())
    for depth, level in by_depth.items():
        step = max_level / max(len(level), 1)
        for idx, node in enumerate(level):
            coords[node.key] = (float(depth), max_level - (idx + 0.5) * step)

    values = [node_metric(node, metric) for node in visible_nodes_list]
    numeric_values = [value for value in values if value is not None]
    low = min([1.0, *numeric_values])
    high = max([1.0, *numeric_values])
    if math.isclose(low, high):
        low -= 0.05
        high += 0.05
    norm = colors.Normalize(vmin=low, vmax=high)
    cmap = cm.get_cmap("RdYlGn")
    best_path = best_path_keys(nodes, ranks)
    side_table = label_mode == "table"
    inline_labels = label_mode == "inline"

    fig_width = max(11.5 if side_table else 10.5, 2.05 * max(depths) + (4.3 if side_table else 0.0))
    fig_height = max(5.5, (0.50 if inline_labels else 0.42) * max_level)
    if side_table:
        fig = plt.figure(figsize=(fig_width, fig_height))
        grid = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.45], wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        table_ax = fig.add_subplot(grid[0, 1])
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        table_ax = None

    for node in visible_nodes_list:
        if node.parent not in visible or node.parent not in coords:
            continue
        x0, y0 = coords[node.parent]
        x1, y1 = coords[node.key]
        edge_in_best_path = node.key in best_path and node.parent in best_path
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="#333333" if edge_in_best_path else "#B8B8B8",
            linewidth=1.6 if edge_in_best_path else 0.75,
            alpha=0.82 if edge_in_best_path else 0.42,
            zorder=2 if edge_in_best_path else 1,
        )

    for node in visible_nodes_list:
        x, y = coords[node.key]
        value = node_metric(node, metric)
        color = "#DDDDDD" if value is None else cmap(norm(value))
        size = 210 + min(520, 90 * math.sqrt(max(node.visits, 1)))
        ax.scatter([x], [y], s=size, c=[color], edgecolors="#222222", linewidths=0.65, zorder=3)
        if inline_labels:
            if label_top <= 0 or node.key == "root" or ranks.get(node.key, math.inf) <= label_top:
                ax.text(
                    x,
                    y,
                    "\n".join(textwrap.wrap(inline_node_text(node, ranks, metric), width=24, break_long_words=False)),
                    ha="center",
                    va="center",
                    fontsize=4.9,
                    color="#111111",
                    zorder=4,
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.36,
                    },
                )
        elif label_mode == "compact":
            ax.text(
                x,
                y,
                node_text(node, ranks, metric),
                ha="center",
                va="center",
                fontsize=6.3,
                color="#111111",
                zorder=4,
            )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(metric.replace("_", " ").replace("speedup", "speedup over baseline"))

    if table_ax is not None:
        table_ax.axis("off")
        table_ax.set_title(f"Top {label_top if label_top > 0 else 'all'} prefixes", fontsize=10, loc="left")
        table_ax.text(
            0.0,
            0.98,
            prefix_table_text(nodes, ranks, metric=metric, limit=label_top),
            ha="left",
            va="top",
            family="monospace",
            fontsize=7.0,
            linespacing=1.35,
        )

    ax.set_title(title)
    ax.set_xlabel("Transform prefix depth")
    ax.set_yticks([])
    ax.set_xticks(depths)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.8)
    ax.set_ylim(-0.8, max_level + 0.8)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot polyMorph MCTS prefix trees ranked by observed performance."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root or config subtree.")
    parser.add_argument("--out-dir", type=Path, default=Path("post_processing/figures/search_trees"))
    parser.add_argument("--benchmark", default="", help="Only plot benchmarks whose path contains this text.")
    parser.add_argument("--max-prefixes", type=int, default=80, help="Maximum ranked prefixes to emphasize. Use 0 for all.")
    parser.add_argument("--max-depth", type=int, default=0, help="Maximum prefix depth. Use 0 for all.")
    parser.add_argument("--metric", choices=["best_speedup", "mean_speedup"], default="best_speedup")
    parser.add_argument(
        "--label-mode",
        choices=["inline", "compact", "table"],
        default="inline",
        help=(
            "inline: small transform labels on nodes; compact: only rank/speedup on nodes; "
            "table: compact nodes plus a ranked prefix table."
        ),
    )
    parser.add_argument(
        "--label-top",
        type=int,
        default=40,
        help="Number of ranked prefixes to label inline or list in the table. Use 0 for all.",
    )
    parser.add_argument("--format", choices=["png", "svg", "pdf"], default="png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    plt, cm, colors = require_matplotlib()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plotted = 0
    for cfg_dir in config_dirs(args.root.resolve()):
        if args.benchmark and args.benchmark not in str(cfg_dir):
            continue
        trials_path = first_existing(cfg_dir, "-trials.csv")
        if not trials_path:
            continue
        nodes = load_prefix_tree(trials_path)
        opt_level, benchmark = opt_and_benchmark(cfg_dir)
        label = f"{opt_level}/{benchmark}"
        out_path = out_dir / f"{slug(label)}_search_tree.{args.format}"
        if draw_tree(
            plt,
            cm,
            colors,
            nodes,
            out_path,
            title=f"{label}: MCTS Prefix Tree",
            metric=args.metric,
            max_prefixes=args.max_prefixes,
            max_depth=args.max_depth,
            label_top=args.label_top,
            label_mode=args.label_mode,
            dpi=args.dpi,
        ):
            plotted += 1
            print(f"Wrote {out_path}")

    if plotted == 0:
        print("No trial CSVs with plottable transform prefixes found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
