#!/usr/bin/env python3
# live_plot_jetson_temps_v2.py
import os, re, time, math, io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path

# matplotlib.use("TkAgg")  # 창이 안 뜨면 주석 해제

CSV_PATH = r"/var/log/jtop_temps.csv"  # 필요 시 절대경로로 수정
SENSORS = ["CPU","CV0","CV1","CV2","GPU","SoC0","SoC1","SoC2","Tj"]
WINDOW_SECONDS = 3600
POLL_INTERVAL = 1.0

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
    return ","

def first_nonempty_line(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                return line.rstrip("\r\n")
    return ""

def parse_line(line: str, sep: str, has_header_decision: bool):
    # return (datetime_obj, dict) or ("HEADER", None) or (None, None)
    parts = [p.strip() for p in re.split(re.escape(sep), line.rstrip("\r\n"))]
    if len(parts) < 2:
        return None, None

    if has_header_decision is None:
        try:
            datetime.fromisoformat(parts[0].replace("Z",""))
        except Exception:
            return "HEADER", None

    try:
        ts_dt = datetime.fromisoformat(parts[0].replace("Z",""))
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
            if v <= -200:
                v = math.nan
        except Exception:
            v = math.nan
        vals[name] = v

    return ts_dt, vals

def main():
    p = Path(CSV_PATH)
    if not p.is_file():
        print(f"CSV not found: {p}")
        return

    sample = first_nonempty_line(CSV_PATH)
    sep = detect_delim(sample) if sample else ","

    last_size = 0
    header_known = None
    times = []  # list[datetime]
    series = {name: [] for name in SENSORS}

    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Jetson Temperature")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    lines = {}
    for name in SENSORS:
        (ln,) = ax.plot([], [], label=name)  # no color specified
        lines[name] = ln
    ax.legend(title="Sensors", loc="best")
    fig.tight_layout()

    print("Live plotting started. Press Ctrl+C to stop.")
    while True:
        try:
            size = p.stat().st_size
            if size < last_size:
                last_size = 0

            with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(last_size)
                chunk = f.read()
                last_size = f.tell()

            if chunk:
                for line in io.StringIO(chunk):
                    if not line.strip():
                        continue
                    ts_dt, vals = parse_line(line, sep, header_known)
                    if ts_dt == "HEADER":
                        header_known = True
                        continue
                    if ts_dt is None:
                        continue
                    if header_known is None:
                        header_known = False

                    times.append(ts_dt)
                    for name in SENSORS:
                        series[name].append(vals[name])

            # window trim
            if times:
                cutoff = times[-1] - timedelta(seconds=WINDOW_SECONDS)
                idx0 = 0
                for i, t in enumerate(times):
                    if t >= cutoff:
                        idx0 = i
                        break
                if idx0 > 0:
                    times[:] = times[idx0:]
                    for name in SENSORS:
                        series[name] = series[name][idx0:]

            # update plot
            if times:
                x = mdates.date2num(times)  # ← epoch2num 대신 date2num
                for name in SENSORS:
                    y = np.array(series[name], dtype=float)
                    lines[name].set_data(x, y)
                ax.set_xlim(x.min(), x.max() if len(x) > 1 else x.min() + 1/86400)  # +1s in days
                ax.relim(visible_only=True)
                ax.autoscale_view(scaley=False)  # keep y 0..100
                plt.pause(0.001)

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break

if __name__ == "__main__":
    main()
