"""Plot V1 out-of-fold errors by target dP, and compare bins by target and predicted pressure."""

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds

DP_BINS = (-np.inf, 30, 40, 50, 60, 80, 100, np.inf)
PREDICTED_DP_BINS = (-np.inf, 50, 60, 80, 100, np.inf)
PRESSURE_BINS = (-np.inf, 900, 920, 940, 960, np.inf)
QUALITY_STYLES = {"Good": ("#0072B2", "o"), "Fair": ("#009E73", "^"), "Poor": ("#D55E00", "s")}


@dataclass(frozen=True)
class Predictions:
    target_dp: np.ndarray
    predicted_dp: np.ndarray
    poci: np.ndarray
    quality: np.ndarray

    @property
    def dp_error(self):
        return self.predicted_dp - self.target_dp

    @property
    def target_pressure(self):
        return self.poci - self.target_dp

    @property
    def predicted_pressure(self):
        return self.poci - self.predicted_dp


def load_predictions(path, samples, catalog):
    verify_catalog_labels(samples, catalog)
    folds = record_folds(len(samples), 5, 1)
    expected = {sample["record"]: fold for sample, fold in zip(samples, folds, strict=True)}
    target, predictions, poci, quality = [], [], [], []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            try:
                record = int(row["record"])
                fold = int(row["fold"])
                prediction = float(row["prediction"])
                cached_target = float(row["target"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Malformed prediction at {path}:{reader.line_num}") from error
            if row["seed"] != "1" or expected.pop(record, None) != fold:
                raise ValueError(f"Unexpected seed, record, or fold for record {record}")
            label = catalog[record]
            if (cached_target != label["dP"] or row["quality"] != label["quality"]
                    or row["quality"] not in QUALITY_STYLES or not np.isfinite(prediction)):
                raise ValueError(f"Invalid prediction or MOTIS label for record {record}")
            environment = label.get("poci")
            if type(environment) not in (float, int) or not np.isfinite(environment):
                raise ValueError(f"Missing or invalid MOTIS POCI for record {record}")
            target.append(label["dP"])
            predictions.append(prediction)
            poci.append(environment)
            quality.append(label["quality"])
    if expected:
        raise ValueError(f"Missing out-of-fold predictions for {len(expected)} records")
    return Predictions(np.asarray(target), np.asarray(predictions), np.asarray(poci), np.asarray(quality))


def summarize_bins(x, residual, edges=DP_BINS):
    """Each bin includes its lower edge and excludes the upper edge."""
    for left, right in pairwise(edges):
        values = residual[(x >= left) & (x < right)]
        if values.size:
            yield left, right, values.size, values.mean(), np.sqrt(np.mean(values**2))


def draw_column(ax, lower, x, residual, quality, *, xlim, edges, tick, xlabel):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    ink, trend, grid = "#172C43", "#633B91", "#E5EAF1"
    handles = []
    for name, (color, marker) in QUALITY_STYLES.items():
        select = quality == name
        handles.append(ax.scatter(x[select], residual[select], s=23, marker=marker, color=color,
                                  alpha=0.58, linewidths=0, zorder=3, label=f"{name}（{select.sum()}）"))
    handles.append(Line2D([], [], color=trend, linewidth=3, label="分箱平均误差"))
    xmin, xmax = xlim
    for panel in (ax, lower):
        panel.set_axisbelow(True)
        panel.grid(axis="y", color=grid, linewidth=0.8)
        panel.tick_params(length=0, pad=7, colors=ink)
        for spine in ("top", "right", "left"):
            panel.spines[spine].set_visible(False)
        panel.spines["bottom"].set_color("#CBD6E2")
        for edge in edges[1:-1]:
            panel.axvline(edge, color=grid, linewidth=0.85, zorder=1)
    ax.axhline(0, color="#697E94", linewidth=1.15, zorder=2)
    ax.set_xlim(xmin, xmax)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    lower.set_xlabel(xlabel)
    lower.yaxis.set_major_locator(MultipleLocator(5))
    lower.xaxis.set_major_locator(MultipleLocator(tick))
    for left, right, count, bias, bin_rmse in summarize_bins(x, residual, edges):
        start, end = max(left, xmin), min(right, xmax)
        mid = (start + end) / 2
        ax.hlines(bias, start + 0.6, end - 0.6, color="white", linewidth=5, zorder=4)
        ax.hlines(bias, start + 0.6, end - 0.6, color=trend, linewidth=2.8, zorder=5)
        lower.hlines(bin_rmse, start + 0.6, end - 0.6, color="#9CAFC3", linewidth=1.2, zorder=2)
        lower.plot(mid, bin_rmse, "o", color=ink, markersize=4, zorder=3)
        lower.annotate(f"{bin_rmse:.1f}", (mid, bin_rmse), xytext=(0, 5), textcoords="offset points",
                       ha="center", va="bottom", color=ink, fontsize=9)
        lower.text(mid, 0.03, f"n={count}", transform=lower.get_xaxis_transform(),
                   ha="center", va="bottom", color="#50647A", fontsize=8)

    return handles


def plot(columns, residual, quality, *, edges, xlim, tick, title, ylabel, note, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    matplotlib.rcParams.update({"font.family": "Noto Sans CJK SC", "font.size": 11, "pdf.fonttype": 42})
    ink, muted = "#172C43", "#50647A"
    side_by_side = len(columns) == 2
    fig, axes = plt.subplots(2, len(columns), figsize=(13.4 if side_by_side else 10.5, 7.1),
                             sharex="col", sharey="row", height_ratios=(3.3, 1.25), squeeze=False)
    left = 0.082 if side_by_side else 0.105
    fig.subplots_adjust(left=left, right=0.975, top=0.745 if side_by_side else 0.775,
                        bottom=0.14, hspace=0.12, wspace=0.075)
    fig.text(left, 0.946, title, fontsize=23, weight="bold", color=ink)
    fig.text(left, 0.898, f"{residual.size} 条 Himawari 记录  ·  记录级五折折外  ·  种子 1", color=muted)
    rmse = np.sqrt(np.mean(residual**2))
    mae = np.mean(np.abs(residual))
    fig.text(0.975, 0.946, f"全体 RMSE  {rmse:.3f} hPa", ha="right", color=ink, weight="bold", fontsize=14)
    fig.text(0.975, 0.898, f"MAE {mae:.3f}  ·  平均误差 {residual.mean():+.3f} hPa", ha="right", color=muted)

    max_rmse = max(row[4] for x, _, _ in columns for row in summarize_bins(x, residual, edges))
    for index, ((x, xlabel, name), (ax, lower)) in enumerate(zip(columns, axes.T, strict=True)):
        handles = draw_column(ax, lower, x, residual, quality, xlim=xlim, edges=edges, tick=tick, xlabel=xlabel)
        if name:
            ax.set_title(name, loc="left", color=ink, fontsize=12, weight="bold")
        if index == 0:
            ax.set_ylabel(ylabel)
            lower.set_ylabel("分箱 RMSE\n（hPa）")
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left - 0.008, 0.862), ncols=4,
                       frameon=False, columnspacing=2.1, handletextpad=0.6)
    axes[1, 0].set_ylim(0, max_rmse + 4)
    fig.text(left, 0.035, note, fontsize=9, color=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_predictions(predictions, catalog_size, output):
    target = predictions.target_dp
    plot(
        [(target, "MOTIS 目标降压 dP（hPa）", "")], predictions.dp_error, predictions.quality,
        edges=DP_BINS, xlim=(min(19, np.floor(target.min()) - 2), max(146, np.ceil(target.max()) + 3)),
        tick=10, title="V1 全体评估记录：误差与降压", ylabel="折外误差（预测 − MOTIS，hPa）",
        note=f"每点一条记录；目录另有 {catalog_size - target.size} 条未进入本次评估。MOTIS 为目录标签，非独立实测。",
        output=output / "v1_error_vs_dp",
    )
    target_pressure = predictions.target_pressure
    predicted_pressure = predictions.predicted_pressure
    plot(
        [(target_pressure, "MOTIS 目录中心气压（hPa）", "按目录气压分箱"),
         (predicted_pressure, "V1 推算中心气压（hPa）", "按推算气压分箱")],
        -predictions.dp_error, predictions.quality, edges=PRESSURE_BINS,
        xlim=(min(860, target_pressure.min() - 4, predicted_pressure.min() - 4),
              max(995, target_pressure.max() + 4, predicted_pressure.max() + 4)),
        tick=20, title="V1 中心气压误差：两种分箱对照", ylabel="气压误差（推算 − 目录，hPa）",
        note="同条记录共用 MOTIS POCI；中心气压 = POCI − dP。对照为目录值，非独立实测。",
        output=output / "v1_error_vs_pressure_bins",
    )


def print_summary(name, key, residual, edges):
    print(f"{name}: r={np.corrcoef(key, residual)[0, 1]:+.3f}")
    print("low high n bias_hPa rmse_hPa")
    for left, right, n, bias, rmse in summarize_bins(key, residual, edges):
        print(f"{left:g} {right:g} {n} {bias:+.3f} {rmse:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        catalog, _ = read_catalog()
        samples = json.loads(args.source_meta.read_text())["records"]
        predictions = load_predictions(args.predictions, samples, catalog)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.error(f"Cannot read V1 evaluation inputs: {error}")
    plot_predictions(predictions, len(catalog), args.output)
    print(f"Saved {predictions.target_dp.size} seed-1 OOF records to {args.output}")
    dp_error = predictions.dp_error
    print_summary("MOTIS dP", predictions.target_dp, dp_error, DP_BINS)
    print_summary("V1 dP", predictions.predicted_dp, dp_error, PREDICTED_DP_BINS)
    print_summary("MOTIS central pressure", predictions.target_pressure, -dp_error, PRESSURE_BINS)
    print_summary("V1 central pressure", predictions.predicted_pressure, -dp_error, PRESSURE_BINS)


if __name__ == "__main__":
    main()
