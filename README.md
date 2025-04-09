# 📱 Action Detection with Spiking Neural Network (SNN)

An efficient gesture detection system using 6-axis sensor data and Spiking Neural Networks.  
Designed for low-power environments such as smartphones.

---

## 📂 Project Structure

```
.
├── data/                      # Sensor data
├── figures/                   # Plots & visualizations
├── model.py                   # SNN model architecture
├── train.py                   # Training script
├── dataset.py                 # Dataset loading and preprocessing
├── spike_encoding.py          # Converts sensor data to spike trains
├── visualize_distribution.py  # Label distribution plot
├── label_gui.py               # GUI tool for time-range labeling
└── ...
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/khyun1109/action_detect_SNN.git
cd action_detect_SNN
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training

```bash
python train.py
```

---

## 📘 For more detail

[👉 Notion Link](https://exclusive-molecule-e71.notion.site/Samsung-MX-HCI-paper-1ced2cb728f08...)

---

## 🧩 Features

- ✅ Gesture classification using SNN
- ✅ 6-axis sensor preprocessing
- ✅ Time-range labeling tool with GUI
- ✅ Compact model for mobile deployment

---

## 📜 License

MIT License
