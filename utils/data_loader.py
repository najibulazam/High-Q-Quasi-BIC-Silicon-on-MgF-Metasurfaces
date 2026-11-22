# utils/data_loader.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class MetasurfaceDataset(Dataset):
    def __init__(self, geometries, spectra):
        self.geometries = torch.tensor(geometries, dtype=torch.float32)
        self.spectra    = torch.tensor(spectra,    dtype=torch.float32)

    def __len__(self):
        return len(self.geometries)

    def __getitem__(self, idx):
        return self.geometries[idx], self.spectra[idx]

def _filter_bad_labels(df, r_cols, edge_margin_frac=0.02, weak_thresh=0.3):
    R = df[r_cols].values
    L = R.shape[1]
    margin = int(edge_margin_frac * L)
    peak_idx = R.argmax(axis=1)
    peak_val = R.max(axis=1)
    is_edge = (peak_idx <= margin) | (peak_idx >= L - 1 - margin)
    is_weak = peak_val < weak_thresh
    bad = is_edge | is_weak
    print(f"  |-- [FILTER] Bad-label samples: {bad.sum()} / {len(df)} ({100*bad.mean():.1f}%) removed")
    return df[~bad].reset_index(drop=True)

def get_meta_loaders():
    csv_path = config.get('dataset', {}).get('output_csv', 'dataset/spectral_data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[DATA]   Loaded CSV: {len(df)} samples | {len(df.columns)} columns from '{csv_path}'")

    geo_cols  = ["H_um", "Px_um", "Py_um", "R1_um", "R2_um"]
    spec_cols = sorted([c for c in df.columns if c.startswith("r_")], key=lambda x: int(x.split("_")[1]))

    df = _filter_bad_labels(df, spec_cols)

    X_geo  = df[geo_cols].values
    Y_spec = df[spec_cols].values

    test_size  = config.get('dataset', {}).get('test_size',  0.20)
    batch_size = config.get('dataset', {}).get('batch_size', 128)

    X_train_geo, X_val_geo, Y_train_spec, Y_val_spec = train_test_split(
        X_geo, Y_spec, test_size=test_size, random_state=42
    )

    scaler_geo = StandardScaler()
    X_train_geo = scaler_geo.fit_transform(X_train_geo)
    X_val_geo   = scaler_geo.transform(X_val_geo)

    spec_mean = Y_train_spec.mean()
    spec_std  = Y_train_spec.std()
    Y_train_spec_n = (Y_train_spec - spec_mean) / (spec_std + 1e-8)
    Y_val_spec_n   = (Y_val_spec   - spec_mean) / (spec_std + 1e-8)
    scaler_spec = {"mean": float(spec_mean), "std": float(spec_std)}
    print(f"  |-- [NORM]  Spectral norm -> mean: {spec_mean:.6f} | std: {spec_std:.6f}")

    train_dataset = MetasurfaceDataset(X_train_geo, Y_train_spec_n)
    val_dataset   = MetasurfaceDataset(X_val_geo,   Y_val_spec_n)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[READY]  Dataset ready -> Train: {len(train_dataset)} | Val: {len(val_dataset)} (Batch: {batch_size})")
    return train_loader, val_loader, scaler_geo, scaler_spec