import pandas as pd
import matplotlib.pyplot as plt

def compare_sensor_with_5ms(original_path: str, path_10ms: str, path_5ms: str, max_points: int = 1000):
    """
    원본 센서 CSV와 10ms/5ms 변환 결과를 함께 시각화합니다.

    Args:
        original_path (str): 원본 CSV 경로
        path_10ms (str): 10ms 변환 결과 CSV 경로
        path_5ms (str): 5ms 변환 결과 CSV 경로
        max_points (int): 그래프에 표시할 최대 포인트 수
    """
    # 원본 데이터
    orig_df = pd.read_csv(original_path)
    orig_df["Timestamp"] = pd.to_datetime(orig_df["Timestamp"], unit="ms")

    orig_acc = orig_df[orig_df["SensorType"] == "ACCELEROMETER"].copy()
    orig_gyro = orig_df[orig_df["SensorType"] == "GYROSCOPE"].copy()
    orig_acc.rename(columns={"X": "ax", "Y": "ay", "Z": "az"}, inplace=True)
    orig_gyro.rename(columns={"X": "gx", "Y": "gy", "Z": "gz"}, inplace=True)

    # 10ms 변환 결과
    df_10ms = pd.read_csv(path_10ms)
    df_10ms["Timestamp"] = pd.to_datetime(df_10ms["Timestamp"], unit="ms")

    # 5ms 변환 결과
    df_5ms = pd.read_csv(path_5ms)
    df_5ms["Timestamp"] = pd.to_datetime(df_5ms["Timestamp"], unit="ms")

    axis_list = ["ax", "ay", "az", "gx", "gy", "gz"]
    plt.figure(figsize=(15, 10))

    for i, axis in enumerate(axis_list):
        plt.subplot(3, 2, i+1)

        # 원본
        if axis.startswith("a") and axis in orig_acc:
            plt.plot(orig_acc["Timestamp"], orig_acc[axis], label="Original", alpha=0.4, linestyle="dotted")
        elif axis in orig_gyro:
            plt.plot(orig_gyro["Timestamp"], orig_gyro[axis], label="Original", alpha=0.4, linestyle="dotted")

        # 10ms
        if axis in df_10ms:
            df = df_10ms[["Timestamp", axis]]
            if len(df) > max_points:
                df = df.iloc[::len(df) // max_points]
            plt.plot(df["Timestamp"], df[axis], label="10ms", linewidth=1.5)

        # 5ms
        if axis in df_5ms:
            df = df_5ms[["Timestamp", axis]]
            if len(df) > max_points:
                df = df.iloc[::len(df) // max_points]
            plt.plot(df["Timestamp"], df[axis], label="5ms", linewidth=1, alpha=0.8)

        plt.title(axis)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Original vs 10ms vs 5ms Sensor Data", fontsize=16, y=1.02)
    plt.savefig("sensor_comparison_5ms_10ms.png")
    print("그래프가 sensor_comparison_5ms_10ms.png 로 저장되었습니다.")

compare_sensor_with_5ms(
    original_path="pluck_up_0.csv",
    path_10ms="pluck_up_0_6axis_ffill_10ms.csv",
    path_5ms="pluck_up_0_6axis_ffill_5ms.csv"
)

