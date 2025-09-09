#!/usr/bin/env python3
import os, sys, re, math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

CANDIDATE_PATHS = [
    "/var/log/jtop_temps.csv",
    os.path.expanduser("~/jtop_logs/jtop_temps.csv"),
]

# 파일에 헤더가 없을 때 사용할 기대 컬럼명(질문에서 주신 순서)
EXPECTED_NAMES = [
    "timestamp", "CPU", "CV0", "CV1", "CV2",
    "GPU", "SoC0", "SoC1", "SoC2", "Tj"
]

def pick_csv():
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            return str(p)
        print(f"CSV not found: {p}")
        sys.exit(1)
    for p in CANDIDATE_PATHS:
        if Path(p).is_file():
            return p
    print("CSV not found in default locations.")
    sys.exit(1)

def looks_like_data_row(line: str) -> bool:
    # ISO-8601 datetime + 숫자들인지 빠르게 판별
    parts = [s.strip() for s in line.strip().split(",")]
    if len(parts) < 2:
        return False
    # 1열: datetime으로 파싱 시도
    try:
        # T가 들어간 ISO도 허용
        _ = datetime.fromisoformat(parts[0].replace("Z",""))
    except Exception:
        return False
    # 2열: 숫자(float) 시도
    try:
        float(parts[1])
    except Exception:
        return False
    return True

def read_csv_auto(path: str) -> pd.DataFrame:
    # 첫 줄 읽어서 헤더 유무 판별
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if looks_like_data_row(first):
        # 헤더 없음
        df = pd.read_csv(path, header=None, names=EXPECTED_NAMES)
    else:
        # 헤더 있음
        df = pd.read_csv(path)
    return df

def main():
    csv_path = pick_csv()
    print(f"Reading: {csv_path}")

    df = read_csv_auto(csv_path)

    if df.shape[1] < 2:
        print("CSV has fewer than 2 columns. Check delimiter or logger.")
        print(df.head())
        sys.exit(1)

    # 시간 컬럼 처리
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)

    # 온도 컬럼만 추출(첫 컬럼 제외)
    temp_cols = [c for c in df.columns if c != time_col]

    # 숫자 변환 및 센서 결측치(-256 등) NaN 처리
    for c in temp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= -200, c] = math.nan  # -256 sentry 제거

    # 전부 NaN인 행 제거
    df = df.dropna(subset=temp_cols, how="all")

    if len(df) == 0:
        print("No valid temperature samples after cleaning.")
        sys.exit(1)

    # 단일 행이면 점, 그 이상이면 라인
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

    # y축 0~100 고정
    ax.set_ylim(0, 100)
    ax.legend(title="Sensors", loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
