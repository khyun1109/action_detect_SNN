import os
import json
import pandas as pd
from glob import glob
from collections import Counter

def create_windows(df, window_size=20, stride=5):
    indices = []
    for start in range(0, len(df) - window_size + 1, stride):
        end = start + window_size
        indices.append((start, end))
    return indices

def auto_label_multiclass(folder, window_size=20, stride=5):
    """
    폴더 내 CSV + ranges.json 파일 기반으로 라벨링:
    - "pluck_up" 포함 → label=1
    - "pluck_down" 포함 → label=2
    - 범위 밖은 모두 label=0
    """
    samples = []

    for csv_path in glob(os.path.join(folder, "*_6axis_ffill.csv")):
        base = os.path.splitext(os.path.basename(csv_path))[0]

        # 라벨 판별
        if "pluck_up" in base:
            label_value = 1
        elif "pluck_down" in base:
            label_value = 2
        else:
            continue

        json_path = os.path.join(folder, f"{base}_ranges.json")
        if not os.path.exists(json_path):
            print(f"⚠️ 누락된 JSON: {json_path}")
            continue

        df = pd.read_csv(csv_path)
        with open(json_path, "r") as f:
            ranges = json.load(f)

        windows = create_windows(df, window_size, stride)

        labels = []
        for start_idx, end_idx in windows:
            ts = df.iloc[start_idx:end_idx]["Timestamp"]
            match = any(((ts >= s) & (ts <= e)).any() for s, e in ranges)
            labels.append(label_value if match else 0)

        print(f"✅ {base}: 클래스 {label_value}, 샘플 수 {len(windows)}, 라벨 분포: {Counter(labels)}")

        samples.append({
            "csv": csv_path,
            "windows": windows,
            "labels": labels,
            "ranges": ranges
        })

    return samples

