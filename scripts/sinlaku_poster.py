"""Build the V1 release poster series and figure for Sinlaku (2026).

Reads the local 10-minute Himawari-9 R304 sequence, runs the exported V1 C++
predictor at 10-minute cadence, and compares the inferred dP with the MOTIS
catalog. Only the MOTIS catalog supplies observed dP, central pressure and POCI;
centres are IBTrACS 3-hourly linear interpolation.
"""

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

BANDS = (8, 9, 10, 16)
# Dataset scans end 10 min before the label time, so the training gaps (300/180/60/30/10/0)
# correspond to these nominal lags. They reproduce the published OOF input exactly.
LAGS = (300, 180, 60, 30, 10, 0)
FOLDS = 5
FIRST = datetime(2026, 4, 11, 5, tzinfo=UTC)
LAST = datetime(2026, 4, 15, 0, tzinfo=UTC)
SID = "2026099N09152"


@dataclass(frozen=True)
class Track:
    seconds: np.ndarray
    lat: np.ndarray
    lon: np.ndarray

    def at(self, when: datetime) -> tuple[float, float]:
        stamp = when.timestamp()
        return float(np.interp(stamp, self.seconds, self.lat)), float(np.interp(stamp, self.seconds, self.lon))


def read_track(path: Path) -> Track:
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        next(rows)
        index = {name: column for column, name in enumerate(header)}
        records = [row for row in rows if len(row) > 3 and row[index["SID"]] == SID]
    records.sort(key=lambda row: row[index["ISO_TIME"]])
    return Track(
        np.array([datetime.strptime(r[index["ISO_TIME"]], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp() for r in records]),
        np.array([float(r[index["LAT"]]) for r in records]),
        np.array([float(r[index["LON"]]) for r in records]),
    )


def catalog_records(path: Path) -> list[tuple[int, dict]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read the MOTIS catalog {path}: {error}") from error
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"The MOTIS catalog {path} has no record list")
    return [(index, record) for index, record in enumerate(records) if record.get("name") == "Sinlaku"]


def scan_files(dataset: Path, when: datetime) -> list[Path]:
    return [
        dataset / f"{when:%Y/%m/%d/%H%M}" / f"HS_H09_{when:%Y%m%d_%H%M}_B{band:02d}_R304_R20_S0101.DAT.bz2"
        for band in BANDS
    ]


def missing_scans(dataset: Path, first: datetime, last: datetime) -> list[str]:
    missing, when = [], first
    while when <= last:
        missing.extend(str(p) for p in scan_files(dataset, when) if not p.is_file())
        when += timedelta(minutes=10)
    return missing


def predict(exe: Path, models: list[Path], dataset: Path, when: datetime,
            centre: tuple[float, float]) -> list[float]:
    names = [scan_files(dataset, when - timedelta(minutes=lag)) for lag in LAGS]
    files = [str(path) for scan in names for path in scan]
    out = []
    for model in models:
        reply = subprocess.check_output([str(exe), f"{centre[0]:.6f}", f"{centre[1]:.6f}", str(model), "--", *files], text=True)
        out.append(float(reply.strip()))
    return out


def next_scan(when: datetime) -> datetime:
    """Round up to the enclosing 10-minute Himawari scan time."""
    base = when.replace(second=0, microsecond=0)
    floor = base.replace(minute=base.minute - base.minute % 10)
    return floor if floor == base else floor + timedelta(minutes=10)


def minutewise(args, track: Track) -> None:
    missing = missing_scans(args.dataset, args.first - timedelta(minutes=max(LAGS)), args.last)
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing scans, first {missing[0]}")
    rows, when = [], args.first
    while when <= args.last:
        rows.append(when)
        when += timedelta(minutes=10)
    centres = [track.at(stamp) for stamp in rows]

    def step(pair: tuple[datetime, tuple[float, float]]) -> list[float]:
        return predict(args.exe, args.models, args.dataset, pair[0], pair[1])

    with ThreadPoolExecutor(args.workers) as pool:
        results = pool.map(step, zip(rows, centres, strict=True))
        steps = [{"time_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "lat": centre[0], "lon": centre[1],
                  **{f"fold{k}": v for k, v in enumerate(values)}, "dP_mean": float(np.mean(values)),
                  "dP_std": float(np.std(values))}
                 for stamp, centre, values in zip(rows, centres, results, strict=True)]
    with args.series.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)
    print(f"wrote {args.series} ({len(steps)} steps)")


def catalog_table(args, track: Track, reference: dict[int, dict]) -> list[dict]:
    rows = []
    for record_index, record in catalog_records(args.catalog):
        when = datetime.strptime(record["time"]["iso_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        if not args.first <= when <= args.last:
            continue
        centre = track.at(when)
        values = predict(args.exe, args.models, args.dataset, next_scan(when), centre)
        held = int(reference[record_index]["fold"])
        rows.append({
            "record": record_index, "time_utc": record["time"]["iso_utc"],
            "anchor_utc": next_scan(when).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": record["location"]["lat"], "lon": record["location"]["lon"],
            "quality": record["quality"], "poci": record["poci"],
            "motis_dp": record["dP"], "motis_pressure": record["pressure"],
            "v1_dp_mean": float(np.mean(values)), "v1_dp_std": float(np.std(values)),
            "v1_dp_held_out": values[held], "v1_fold": held,
            "v1_pressure_mean": record["poci"] - float(np.mean(values)),
            "v1_pressure_held_out": record["poci"] - values[held],
            "published_oof_dp": float(reference[record_index]["prediction"]),
        })
    return rows


def figure(args, series: list[dict], catalog: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": ["Noto Serif CJK SC", "DejaVu Serif"], "font.size": 9,
                         "axes.unicode_minus": False, "axes.grid": True, "grid.alpha": 0.25})

    def stamps(rows):
        return [datetime.strptime(r["time_utc"], "%Y-%m-%dT%H:%M:%SZ") for r in rows]

    time = stamps(series)
    dp = np.array([float(r["dP_mean"]) for r in series])
    dp_std = np.array([float(r["dP_std"]) for r in series])
    cat_time = stamps(catalog)
    poci = np.array([float(r["poci"]) for r in catalog])
    poci_line = np.interp([t.timestamp() for t in time], [t.timestamp() for t in cat_time], poci)
    pressure = poci_line - dp

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True, gridspec_kw={"hspace": 0.16})
    top.fill_between(time, dp - dp_std, dp + dp_std, color="#1f6fb4", alpha=0.18, linewidth=0,
                     label="V1 五折集成 ±1σ")
    top.plot(time, dp, color="#1f6fb4", linewidth=1.4, label="V1 推算 dP（10 min）")
    top.plot(cat_time, [float(r["motis_dp"]) for r in catalog], "o", color="#c0392b", markersize=5,
             markerfacecolor="white", markeredgewidth=1.3, label="MOTIS 实测 dP（9 条记录）")
    top.set_ylabel("降压 dP（hPa）")
    top.legend(loc="lower right", frameon=False)

    bottom.fill_between(time, pressure - dp_std, pressure + dp_std, color="#2e8b57", alpha=0.16, linewidth=0,
                        label="V1 五折集成 ±1σ")
    bottom.plot(time, pressure, color="#2e8b57", linewidth=1.4, label="推算中心气压 = POCI − dP")
    bottom.plot(cat_time, [float(r["motis_pressure"]) for r in catalog], "o", color="#c0392b", markersize=5,
                markerfacecolor="white", markeredgewidth=1.3, label="MOTIS 实测气压（POCI − dP）")
    bottom.set_ylabel("中心气压（hPa）")
    bottom.legend(loc="lower left", frameon=False)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=UTC))
    bottom.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12), tz=UTC))
    bottom.set_xlabel("UTC")

    errors = np.array([float(r["v1_dp_mean"]) - float(r["motis_dp"]) for r in catalog])
    rmse = float(np.sqrt(np.mean(errors**2)))
    fig.suptitle("V1 Release｜Sinlaku（2026）2026-04-11 05:00Z – 2026-04-15 00:00Z", fontsize=12)
    lines = [
        "推理 inference_best.cpp：V1 五折等权集成，每步 6 帧（−300/−180/−60/−30/−10/0 min）B08/B09/B10/B16 R304 2 km；"
        + f"中心为 IBTrACS 三小时线性插值。目录时刻 dP 平均误差 {errors.mean():+.2f} hPa、RMSE {rmse:.2f} hPa。",
        "POCI 取 MOTIS 目录值线性插值；下图为 POCI − dP 的模型推算气压，不是实测中心气压。",
    ]
    fig.text(0.5, 0.012, "\n".join(lines), ha="center", va="bottom", fontsize=7.2, color="#444444", linespacing=1.5)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.13)
    for suffix in ("png", "pdf"):
        fig.savefig(args.figure.with_suffix("." + suffix), dpi=300)
    print("wrote", args.figure.with_suffix(".png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("MOTIS_cyclone_records.json"))
    parser.add_argument("--ibtracs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="V1 formal/predictions.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--first", type=_time, default=FIRST)
    parser.add_argument("--last", type=_time, default=LAST)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--reuse-series", action="store_true", help="Plot the existing series file")
    args = parser.parse_args()
    args.models = [args.models / f"fold{k}.v1m" for k in range(FOLDS)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)

    track = read_track(args.ibtracs)
    with args.reference.open(newline="") as stream:
        reference = {int(row["record"]): row for row in csv.DictReader(stream)}
    minutewise(args, track) if not args.reuse_series else print(f"reusing {args.series}")
    with args.series.open(newline="") as stream:
        series = list(csv.DictReader(stream))
    catalog = catalog_table(args, track, reference)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    print(f"wrote {args.output} ({len(catalog)} catalog records)")
    figure(args, series, catalog)


def _time(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


if __name__ == "__main__":
    main()
