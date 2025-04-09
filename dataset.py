import os
import json
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset, DataLoader
from spike_encoding import rate_code_zscore_sigmoid

class SpikeDataset(Dataset):
    def __init__(self, csv_path, windows, labels, spike_func, T=20):
        self.df = pd.read_csv(csv_path)
        self.windows = windows
        self.labels = labels
        self.spike_func = spike_func
        self.T = T

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start, end = self.windows[idx]
        df_win = self.df.iloc[start:end].reset_index(drop=True)
        spike_tensor, _ = self.spike_func(df_win, T=self.T)  # shape: (N, 12, T)
        spike_tensor = spike_tensor.transpose(1, 0, 2)  # → (12, N, T)
        return torch.tensor(spike_tensor, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def create_datasets_from_samples(samples, spike_func, T=20):
    """
    auto_label_all() 결과 리스트에서 PyTorch Dataset 리스트 생성

    Returns:
        List[SpikeDataset]
    """
    datasets = []

    for sample in samples:
        dataset = SpikeDataset(
            csv_path=sample["csv"],
            windows=sample["windows"],
            labels=sample["labels"],
            spike_func=spike_func,
            T=T
        )
        datasets.append(dataset)
        print(f"📦 Dataset 생성됨: {os.path.basename(sample['csv'])}, 샘플 수: {len(dataset)}")

    return datasets
