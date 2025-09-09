#!/usr/bin/env python3
import csv, time, datetime, os
from jtop import jtop

OUTFILE = "/var/log/jtop_temps.csv"   # 경로 변경 가능
INTERVAL_S = 1.0                      # 샘플링 주기 [s]

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def main():
    ensure_dir(OUTFILE)
    with jtop() as jetson:
        # 첫 샘플에서 센서 목록 파악
        stats = jetson.stats
        temp_map = stats.get("temp", {})
        # 필드명: timestamp + temp.<sensor>
        fieldnames = ["timestamp"] + [f"temp.{k}" for k in sorted(temp_map.keys())]

        # 파일이 없으면 헤더 생성, 있으면 append
        file_exists = os.path.isfile(OUTFILE)
        with open(OUTFILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            while jetson.ok():
                stats = jetson.stats
                temp_map = stats.get("temp", {})

                row = {"timestamp": datetime.datetime.now().isoformat()}
                for k in temp_map.keys():
                    row[f"temp.{k}"] = temp_map.get(k)

                writer.writerow(row)
                f.flush()
                time.sleep(INTERVAL_S)

if __name__ == "__main__":
    main()
