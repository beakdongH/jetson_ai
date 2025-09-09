# Robust CSV loader and plotter for Jetson temperature logs
# - Auto-detect delimiter (comma, fullwidth comma, tab, semicolon, pipe)
# - Handle headerless files (assign expected names)
# - Convert -256-like sentinels to NaN
# - Plot with matplotlib (single plot, no colors set)
# - Y-axis fixed to 0..100 °C
# - Display a small preview of parsed data for verification

import re, math, os, io, sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from caas_jupyter_tools import display_dataframe_to_user

CSV_PATH = "/mnt/data/jtop_temps.csv"

EXPECTED_NAMES = [
    "timestamp","CPU","CV0","CV1","CV2","GPU","SoC0","SoC1","SoC2","Tj"
]

DELIMS = [
    (",", "ASCII comma"),
    ("，", "Fullwidth comma"),
    ("\t", "Tab"),
    (";", "Semicolon"),
    ("|", "Pipe"),
]

def first_nonempty_line(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                return line.rstrip("\r\n")
    return ""

def detect_delim(sample_line: str):
    counts = {name: sample_line.count(ch) for ch, name in DELIMS}
    chosen_name, chosen_count = max(counts.items(), key=lambda kv: kv[1])
    delim_char = None
    for ch, name in DELIMS:
        if name == chosen_name:
            delim_char = ch
            break
    return delim_char, chosen_name, counts

def read_df(path, sep_char):
    # Determine if header exists by checking first field as ISO datetime, second as numeric
    line = first_nonempty_line(path)
    parts = [p.strip() for p in re.split(re.escape(sep_char), line)]
    headerless = False
    if parts:
        try:
            _ = datetime.fromisoformat(parts[0].replace("Z",""))
            if len(parts) > 1:
                _ = float(parts[1].replace("+","").replace(",","."))
                headerless = True
        except Exception:
            headerless = False

    if headerless:
        df = pd.read_csv(path, header=None, names=EXPECTED_NAMES,
                         sep=re.escape(sep_char), engine="python")
    else:
        df = pd.read_csv(path, sep=re.escape(sep_char), engine="python")
        # If it looks like 10 columns but wrong names, normalize
        if df.shape[1] == 10 and df.columns[0] != "timestamp":
            df.columns = EXPECTED_NAMES
    return df

def clean_df(df):
    if df.shape[1] < 2:
        return df, ["<insufficient>"]

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)

    temp_cols = [c for c in df.columns if c != time_col]

    # Convert to numeric and remove sentinels (<= -200)
    for c in temp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -200, c] = math.nan

    # Drop rows with all NaN temps
    df = df.dropna(subset=temp_cols, how="all")
    return df, temp_cols

def main():
    path = Path(CSV_PATH)
    if not path.is_file():
        print(f"CSV not found at {CSV_PATH}")
        raise SystemExit(1)

    sample = first_nonempty_line(CSV_PATH)
    if not sample:
        print("CSV is empty.")
        raise SystemExit(1)

    delim_char, delim_name, counts = detect_delim(sample)
    print("DETECT REPORT")
    for ch, name in DELIMS:
        print(f"  {name}: {counts[name]}")
    print(f"  -> chosen: {delim_name} ({'U+%04X' % ord(delim_char)})")

    df_raw = read_df(CSV_PATH, delim_char)
    df, temp_cols = clean_df(df_raw)

    # Show preview to user
    preview = df.head(10).copy()
    display_dataframe_to_user("Parsed temperature preview", preview.reset_index())

    if df.shape[1] < 1 or len(temp_cols) < 1 or len(df) == 0:
        print("After parsing and cleaning, no plottable data was found.")
        print("Head (raw):")
        print(df_raw.head())
        raise SystemExit(0)

    # Plot
    if len(df) == 1:
        ax = df[temp_cols].plot(
            style="o",
            title="Jetson Temperature",
            ylabel="Temperature (°C)",
            xlabel="Time",
        )
    else:
        ax = df[temp_cols].plot(
            title="Jetson Temperature",
            ylabel="Temperature (°C)",
            xlabel="Time",
        )
    ax.set_ylim(0, 100)
    ax.legend(title="Sensors", loc="best")
    plt.tight_layout()
    plt.show()

main()
