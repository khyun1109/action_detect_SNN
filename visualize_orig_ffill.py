import pandas as pd
import matplotlib.pyplot as plt

def compare_sensor_csvs(original_path: str, transformed_path: str, max_points: int = 1000):
    """
    원본 CSV와 변환된 6축 CSV를 시각적으로 비교합니다.

    Args:
        original_path (str): 원본 CSV 파일 경로
        transformed_path (str): 변환된 CSV 파일 경로
        max_points (int): 시각화를 위한 최대 포인트 수 (너무 크면 느림)
    """
    # 원본 파일 읽기
    orig_df = pd.read_csv(original_path)
    trans_df = pd.read_csv(transformed_path)

    # 시간 축 변환
    orig_df["Timestamp"] = pd.to_datetime(orig_df["Timestamp"], unit="ms")
    trans_df["Timestamp"] = pd.to_datetime(trans_df["Timestamp"], unit="ms")

    # 가속도계
    orig_acc = orig_df[orig_df["SensorType"] == "ACCELEROMETER"].copy()
    orig_acc.rename(columns={"X": "ax", "Y": "ay", "Z": "az"}, inplace=True)

    # 자이로스코프
    orig_gyro = orig_df[orig_df["SensorType"] == "GYROSCOPE"].copy()
    orig_gyro.rename(columns={"X": "gx", "Y": "gy", "Z": "gz"}, inplace=True)

    # 시각화
    plt.figure(figsize=(15, 10))

    for i, axis in enumerate(["ax", "ay", "az", "gx", "gy", "gz"]):
        plt.subplot(3, 2, i+1)
        if axis.startswith("a"):
            if axis in orig_acc:
                plt.plot(orig_acc["Timestamp"], orig_acc[axis], label="Original", alpha=0.5, linestyle='dotted')
        else:
            if axis in orig_gyro:
                plt.plot(orig_gyro["Timestamp"], orig_gyro[axis], label="Original", alpha=0.5, linestyle='dotted')

        if axis in trans_df:
            # 너무 많으면 샘플링
            data = trans_df[["Timestamp", axis]]
            if len(data) > max_points:
                data = data.iloc[::len(data) // max_points]

            plt.plot(data["Timestamp"], data[axis], label="Transformed", alpha=0.8)

        plt.title(axis)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Original vs Transformed Sensor Data", fontsize=16, y=1.02)

    # GUI가 없으므로 파일로 저장
    plt.savefig("sensor_comparison.png")

compare_sensor_csvs("pluck_up_0.csv", "pluck_up_0_6axis_ffill.csv")

