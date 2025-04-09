import pandas as pd
import matplotlib.pyplot as plt
import json
import os

class InteractiveLabelTool:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.timestamps = self.df["Timestamp"]
        self.ax = self.df["ax"]
        self.ranges = []
        self.start_ts = None
        self.csv_path = csv_path

    def on_click(self, event):
        if event.inaxes:
            # x값은 Timestamp로 간주
            click_ts = int(event.xdata)

            if self.start_ts is None:
                self.start_ts = click_ts
                print(f"[Start] {self.start_ts}")
            else:
                end_ts = click_ts
                if end_ts < self.start_ts:
                    self.start_ts, end_ts = end_ts, self.start_ts
                else:
                    start_ts = self.start_ts

                self.ranges.append((start_ts, end_ts))
                print(f"[Select Range] {start_ts} ~ {end_ts}")

                self._highlight_range(start_ts, end_ts)
                self.start_ts = None

    def _highlight_range(self, start_ts, end_ts):
        if start_ts is None or end_ts is None:
            return

        ax = plt.gca()
        ax.axvspan(start_ts, end_ts, ymin=0, ymax=1, color="orange", alpha=0.3)
        plt.draw()

    def run(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.timestamps, self.ax, label="ax")
        ax.set_title("Click → Select operation range (2 click = 1range)")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("ax Value")
        plt.grid(True)
        plt.legend()

        fig.canvas.mpl_connect("button_press_event", self.on_click)
        plt.show()

        print("\n✅ Selected Label range:")
        for start, end in self.ranges:
            print(f"({start}, {end})")

        return self.ranges

    def save_range_to_json(self, output_path=None):
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
            output_path = f"{base_name}_ranges.json"

        with open(output_path, "w") as f:
            json.dump(self.ranges, f, indent=4)

        print(f"Label ranges is stored at {output_path}")

tool = InteractiveLabelTool("./data/pluck_down_2_6axis_ffill.csv")
selected_ranges = tool.run()

tool.save_range_to_json()
