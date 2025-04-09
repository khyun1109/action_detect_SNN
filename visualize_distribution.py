import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sensor_distribution(csv_path: str, columns=("ax", "ay", "az", "gx", "gy", "gz")):
    """
    6축 센서 데이터에 대해 각 축의 값 분포를 히스토그램 + KDE로 시각화합니다.

    Args:
        csv_path (str): 6축 CSV 경로 (axis_converter 결과물)
        columns (tuple): 분포를 볼 컬럼들 (기본: 6축)
    """
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(columns):
        if col in df.columns:
            plt.subplot(2, 3, i + 1)
            sns.histplot(df[col], bins=50, kde=True, color="steelblue")
            plt.title(f"{col} Distribution")
            plt.xlabel(col)
            plt.grid(True)

    plt.tight_layout()
    plt.suptitle("Sensor Value Distributions", fontsize=16, y=1.03)
    plt.savefig("./figures/sensor_distribution.png")
    print("분포 그래프가 sensor_distribution.png 로 저장되었습니다.")

plot_sensor_distribution("pluck_down_0_6axis_ffill.csv")

