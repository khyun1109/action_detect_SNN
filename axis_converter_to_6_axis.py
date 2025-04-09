import pandas as pd
import os

def convert_to_6axis_ffill(csv_path: str, output_path: str = None, resample_interval: str = "10ms"):
    """
    센서 데이터를 10ms 간격으로 리샘플링하고, 이전 값이 없으면 forward-fill 방식으로 보간하여 6축으로 병합

    Args:
        csv_path (str): 입력 CSV 경로
        output_path (str): 저장할 CSV 경로 (None이면 자동 생성)
        resample_interval (str): 리샘플링 간격 (기본: 10ms)

    Returns:
        pd.DataFrame: 변환된 6축 센서 데이터
    """
    df = pd.read_csv(csv_path)

    # 센서 분리
    accel_df = df[df["SensorType"] == "ACCELEROMETER"].copy()
    gyro_df = df[df["SensorType"] == "GYROSCOPE"].copy()

    accel_df = accel_df[["Timestamp", "X", "Y", "Z"]].rename(columns={"X": "ax", "Y": "ay", "Z": "az"})
    gyro_df = gyro_df[["Timestamp", "X", "Y", "Z"]].rename(columns={"X": "gx", "Y": "gy", "Z": "gz"})

    # datetime 인덱스 변환
    accel_df["Timestamp"] = pd.to_datetime(accel_df["Timestamp"], unit="ms")
    gyro_df["Timestamp"] = pd.to_datetime(gyro_df["Timestamp"], unit="ms")

    accel_df.set_index("Timestamp", inplace=True)
    gyro_df.set_index("Timestamp", inplace=True)

    accel_df = accel_df[~accel_df.index.duplicated(keep='first')]
    gyro_df = gyro_df[~gyro_df.index.duplicated(keep='first')]

    # 리샘플링 + 이전 값 유지
    resampled_accel = accel_df.resample(resample_interval).ffill()
    resampled_gyro = gyro_df.resample(resample_interval).ffill()

    # 공통 타임스탬프 기준으로 정렬
    common_index = resampled_accel.index.union(resampled_gyro.index).sort_values()
    resampled_accel = resampled_accel.reindex(common_index).ffill()
    resampled_gyro = resampled_gyro.reindex(common_index).ffill()

    # 병합
    merged_df = pd.concat([resampled_accel, resampled_gyro], axis=1)

    # 완전히 비어 있는 행 제거
    merged_df = merged_df.dropna()

    # Timestamp 열 복원
    merged_df = merged_df.reset_index()
    merged_df["Timestamp"] = merged_df["Timestamp"].astype("int64") // 1_000_000  # ms

    # 저장
    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = base + "_6axis_ffill.csv"

    merged_df.to_csv(output_path, index=False)
    print(f"[✓] Forward-fill 방식 변환 완료: {output_path}")

    return merged_df

df_6axis = convert_to_6axis_ffill("./data/pluck_down_2.csv")
