#!/usr/bin/env python3
# live_plot_jetson_temps.py
# Real-time plotting for Jetson temperature CSV appended by your logger.
# - Single axes, multiple lines (one per sensor)
# - No seaborn, no color specification
# - Labels in English
# - y-axis fixed to 0..100 °C
# - -256 (or <= -200) treated as NaN
# - Robust to header/no-header and common delimiters

import os, re, time, math, io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

# VSCode 환경에서 창이 안 뜨면 이 줄의 주석을 해제하세요 (OS에 python3-tk 필요).
# matplotlib.use("TkAgg")

CSV_PATH = r"/var/log/jtop_temps.csv"  # 필요 시 절대경로로 수정 (예: r"C:\path\to\jtop_temps.csv")
# CSV_PATH = r"/home/USER/jtop_logs/jtop_temps.csv"

SENSORS = ["CPU","CV0","CV1","CV2","GPU","SoC0","SoC1","SoC2","Tj"]
EXPECTED_COLS = ["timestamp"] + SENSORS

# Plot window: 최근 N초만 유지 (메모리/성능 관리용)
WINDOW_SECONDS = 3600  # 1시간
POLL_INTERVAL = 1.0    # 초

DELIMS = [
    (",", "ASCII comma"),
    ("，","Fullwidth comma"),
    ("\t","Tab"),
    (";","Semicolon"),
    ("|","Pipe"),
]

def detect_delim(sample_line: str) -> str:
    counts = {name: sample_line.count(ch) for ch, name in DELIMS}
    name, _ = max(counts.items(), key=lambda kv: kv[1])
    for ch, nm in DELIMS:
        if nm == name:
            return ch
    return ","  # fallback

def first_nonempty_line(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                return line.rstrip("\r\n")
    return ""

def parse_line(line: str, sep: str, has_header_decision: bool):
    # returns (ts_float, values_dict) or (None, None) if unparsable/header
    parts = [p.strip() for p in re.split(re.escape(sep), line.rstrip("\r\n"))]
    if len(parts) < 2:
        return None, None

    # header detection: if first token contains non-digit for date and decision == None
    if has_header_decision is None:
        try:
            datetime.fromisoformat(parts[0].replace("Z",""))
            # first token is datetime -> it's data
            pass
        except Exception:
            # header
            return "HEADER", None

    # parse timestamp
    try:
        ts = datetime.fromisoformat(parts[0].replace("Z",""))
    except Exception:
        return None, None

    vals = {}
    for i, name in enumerate(SENSORS, start=1):
        if i >= len(parts):
            vals[name] = math.nan
            continue
        token = parts[i].replace("+","").replace(",",".")
        try:
            v = float(token)
            if v <= -200:  # -256 sentry etc.
                v = math.nan
        except Exception:
            v = math.nan
        vals[name] = v

    return ts.timestamp(), vals

def main():
    p = Path(CSV_PATH)
    if not p.is_file():
        print(f"CSV not found: {p}")
        return

    # delimiter detection
    sample = first_nonempty_line(CSV_PATH)
    if not sample:
        print("CSV is empty; waiting for data...")
        sep = ","
    else:
        sep = detect_delim(sample)

    # state
    last_size = 0
    header_known = None  # None: unknown, True: has header already skipped, False: no header
    times = []  # epoch seconds
    series = {name: [] for name in SENSORS}

    # figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Jetson Temperature")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    lines = {}
    for name in SENSORS:
        # create one line per sensor; no color specified
        (ln,) = ax.plot([], [], label=name)
        lines[name] = ln
    ax.legend(title="Sensors", loc="best")
    fig.tight_layout()

    print("Live plotting started. Press Ctrl+C to stop.")
    while True:
        try:
            size = p.stat().st_size
            # log rotation/truncation handling
            if size < last_size:
                last_size = 0
            # read new bytes
            with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_size)
                chunk = f.read()
                last_size = f.tell()

            if chunk:
                for line in io.StringIO(chunk):
                    if not line.strip():
                        continue
                    ts, vals = parse_line(line, sep, header_known)
                    if ts == "HEADER":
                        header_known = True
                        continue
                    if ts is None:
                        continue
                    if header_known is None:
                        header_known = False

                    times.append(ts)
                    for name in SENSORS:
                        series[name].append(vals[name])

            # drop old points outside window
            if times:
                cutoff = times[-1] - WINDOW_SECONDS
                # find first index >= cutoff
                idx0 = 0
                for i, t in enumerate(times):
                    if t >= cutoff:
                        idx0 = i
                        break
                if idx0 > 0:
                    times[:] = times[idx0:]
                    for name in SENSORS:
                        series[name] = series[name][idx0:]

            # update plot if we have data
            if times:
                x = mdates.epoch2num(np.array(times, dtype=float))
                for name in SENSORS:
                    y = np.array(series[name], dtype=float)
                    # Update line data
                    lines[name].set_data(x, y)
                # set x-limits to data range
                ax.set_xlim(x.min(), x.max() if x.size > 1 else x.min() + 1/86400)  # +1s in days
                ax.relim(visible_only=True)
                ax.autoscale_view(scaley=False)  # y fixed (0..100)

                plt.pause(0.001)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break

if __name__ == "__main__":
    main()
