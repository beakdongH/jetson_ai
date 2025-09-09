#!/usr/bin/env python3
# plot_jetson_temps_vs.py
# - CSV: timestamp,CPU,CV0,CV1,CV2,GPU,SoC0,SoC1,SoC2,Tj (헤더 없다고 가정)
# - delimiter 자동 감지(콤마/전각 콤마/탭/세미콜론/파이프)
# - -256 이하 sentry는 NaN 처리
# - y-axis 0..100°C
# - 창 표시 + PNG 저장

import os, re, math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 필요시 GUI 백엔드 지정 (창이 안 뜨면 주석 해제)
# matplotlib.use("TkAgg")

CSV_PATH = r"/absolute/path/to/jtop_temps.csv"  # ← VSCode PC의 실제 절대경로로 바꾸세요

EXPECTED_NAMES = [
    "timestamp","CPU","CV0","CV1","CV2","GPU","SoC0","SoC1","SoC2","Tj"
]
DELIMS = [(",", "ASCII comma"), ("，","Fullwidth comma"), ("\t","Tab"), (";","Semicolon"), ("|","Pipe")]

def first_nonempty_line(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                return line.rstrip("\r\n")
    return ""

def detect_delim(sample: str):
    counts = {name: sample.count(ch) for ch, name in DELIMS}
    chosen = max(counts.items(), key=lambda kv: kv[1])[0]
    delim_char = [ch for ch, name in DELIMS if name == chosen][0]
    return delim_char, chosen, counts

def read_df(path: str, sep_char: str):
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
        if df.shape[1] == 10 and df.columns[0] != "timestamp":
            df.columns = EXPECTED_NAMES
    return df

def main():
    p = Path(CSV_PATH)
    if not p.is_file():
        print(f'CSV not found: {p}')
        return

    sample = first_nonempty_line(CSV_PATH)
    if not sample:
        print("CSV is empty.")
        return

    sep_char, sep_name, counts = detect_delim(sample)
    print("Delimiter detect:", {k:v for k,v in counts.items()}, "->", sep_name)

    df = read_df(CSV_PATH, sep_char)
    print(f"Data shape raw: {df.shape}")

    if df.shape[1] < 2:
        print("Parsed fewer than 2 columns. Check delimiter or CSV content.")
        print(df.head())
        return

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)

    temp_cols = [c for c in df.columns if c != time_col]
    for c in temp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -200, c] = math.nan
    df = df.dropna(subset=temp_cols, how="all")
    print(f"Data rows after cleaning: {len(df)}")

    if len(df) == 0:
        print("No valid temperature samples after cleaning.")
        return

    if len(df) == 1:
        ax = df[temp_cols].plot(style="o", title="Jetson Temperature",
                                ylabel="Temperature (°C)", xlabel="Time")
    else:
        ax = df[temp_cols].plot(title="Jetson Temperature",
                                ylabel="Temperature (°C)", xlabel="Time")

    ax.set_ylim(0, 100)
    ax.legend(title="Sensors", loc="best")
    plt.tight_layout()

    # 창 표시
    plt.show()

    # PNG 저장(창이 안 떠도 파일로 결과 확인 가능)
    out_png = p.with_suffix(".png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
