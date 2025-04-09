import numpy as np
import pandas as pd
from scipy.special import expit

def rate_code_zscore_sigmoid(df: pd.DataFrame, T: int = 20):
    assert all(col in df.columns for col in ["ax", "ay", "az", "gx", "gy", "gz"]), "6-axis sensor column is needed."

    channels = []
    spikes = []

    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        signal = df[col].to_numpy()

        pos = np.clip(signal, 0, None)
        neg = np.clip(-signal, 0, None)

        for s, suffix in zip([pos, neg], ["_pos", "_neg"]):
            # z-score normalization
            mean = np.mean(s)
            std = np.std(s) + 1e-8
            z = (s - mean) / std

            # sigmoid
            prob = expit(z)

            # rate coding
            spike_seq = (np.random.rand(len(s), T) < prob[:, None]).astype(np.uint8)

            spikes.append(spike_seq)
            channels.append(col + suffix)

    spike_tensor = np.stack(spikes, axis=1)
    return spike_tensor, channels
