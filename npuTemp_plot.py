#!/usr/bin/env python3
import os, re, sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CANDIDATE_PATHS = [
    "/var/log/jtop_temps.csv",
    os.path.expanduser("~/jtop_logs/jtop_temps.csv"),
]

def pick_csv():
    # 명시 경로 인자 우선
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_file():
            return str(p)
        else:
            print(f'CSV not found: {p}')
            sys.exit(1)
    # 후보 경로 자동 선택
    for p in CANDIDATE_PATHS:
        if Path(p).is_file():
            return p
    print("CSV not found in default locations.")
    sys.exit(1)

def englishize(name: str) -> str:
    n = name.strip()
    low = n.lower()

    # 공백/기호 정리
    n = re.sub(r"\s+", "", n)

    # 패턴 매핑
    if "cpu" in low:
        return "CPU"
    if "gpu" in low:
        return "GPU"
    if "tj" in low:
        return "Tj"
    # cv0, cv1, ...
    m = re.match(r".*cv\s*([0-9]+).*", low)
    if not m:
        m = re.match(r".*cv([0-9]+).*", low)
    if m:
        return f"CV{m.group(1)}"
    # soc0, soc1, ...
    m = re.match(r".*soc\s*([0-9]+).*", low)
    if not m:
        m = re.match(r".*soc([0-9]+).*", low)
    if m:
        return f"SoC{m.group(1)}"

    # 일반 fallback: 영문/숫자만 남기기
    n2 = re.sub(r"[^A-Za-z0-9]", "", n)
    return n2 if n2 else name

def main():
    csv_path = pick_csv()
    print(f"Reading: {csv_path}")

    # 헤더 자동 인식, 첫 컬럼을 시간으로 처리
    # encoding은 기본으로 시도, 필요시 encoding='utf-8-sig' 또는 'cp949'로 바꿔보세요.
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        print("CSV must have at least two columns (time + temperatures).")
        sys.exit(1)

    # 첫 컬럼을 시간
    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col)

    # 온도 컬럼만 선별: 숫자형이고, 시간 아닌 것
    # 또한 사용자가 말한 패턴(cpu, gpu, cv, soc, tj)을 우선 포함
    temp_cols = []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ["cpu", "gpu", "cv", "soc", "tj", "온도"]):
            temp_cols.append(c)
    # 혹시 위 패턴이 없더라도 숫자형이면 후보
    for c in df.columns:
        if c not in temp_cols:
            try:
                pd.to_numeric(df[c])
                temp_cols.append(c)
            except Exception:
                pass

    # 중복 제거, 시간 컬럼 제외
    temp_cols = [c for c in dict.fromkeys(temp_cols) if c != time_col]

    if not temp_cols:
        print("No temperature columns detected.")
        sys.exit(1)

    # 숫자 변환
    for c in temp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=temp_cols, how="all")

    # 과도한 포인트면 decimation (예: 20000포인트 초과 시 5배 간격 샘플링)
    N = len(df)
    step = 1
    if N > 20000:
        step = 5
    dfp = df.iloc[::step, :]

    # 범례에 영어 표기 사용하도록 컬럼명 매핑
    rename_map = {c: englishize(c) for c in temp_cols}
    dfp = dfp.rename(columns=rename_map)
    plot_cols = list(rename_map.values())

    # 단일 플롯(여러 라인)
    ax = dfp[plot_cols].plot(
        title="Jetson Temperature",
        ylabel="Temperature (°C)",
        xlabel="Time",
    )
    ax.legend(title="Sensors", loc="best")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
