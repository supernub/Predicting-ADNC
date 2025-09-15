#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a cell-level MLP to classify ADNC (4 classes) using donor-level ADNC labels
assigned to each cell. Optionally filter by one cell type and/or use scVI embeddings.

Usage:
  python cell_adnc_mlp.py \
    --train-h5ad ./data/split_train/cell_type_Astrocyte.h5ad \
    --test-h5ad  ./data/split_test/cell_type_Astrocyte.h5ad \
    --use-scvi \
    --epochs 10 --batch-size 128

Notes:
- ADNC classes: ["Not AD","Low","Intermediate","High"] -> [0,1,2,3]
- Default input = adata.X (log-normalized CPM/10k). Add --use-scvi to use obsm['X_scVI'].
- You can filter a specific cell type by providing the pre-split h5ad (as you already do),
  or use --cell-type to filter inside the script (expects obs['Supertype']).
"""

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# ---------------------- Args ----------------------
ap = argparse.ArgumentParser()
ap.add_argument("--train-h5ad", required=True, help="path to train .h5ad")
ap.add_argument("--test-h5ad",  required=True, help="path to test .h5ad")
ap.add_argument("--cell-type",  default=None, help="optional Supertype filter, e.g., 'Astrocyte'")
ap.add_argument("--label-col",  default="ADNC", help="obs column name for ADNC")
ap.add_argument("--use-scvi",   action="store_true", help="use obsm['X_scVI'] as features")
ap.add_argument("--epochs",     type=int, default=10)
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--lr",         type=float, default=1e-4)
ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--device",     default=None, help="e.g., cuda:0 / cuda:1 / cpu (auto if not set)")
ap.add_argument("--max-cells-per-donor", type=int, default=None,
                help="optional cap per donor to avoid class/donor imbalance & OOM")
ap.add_argument("--outdir",     default="output_models/MLP_ADNC")
ap.add_argument("--save-every", type=int, default=5, help="save every N epochs")
args = ap.parse_args()

# ---------------------- Label mapping ----------------------
ADNC_MAP = {"Not AD":0, "Low":1, "Intermediate":2, "High":3}
ADNC_SET = set(ADNC_MAP.keys())
NUM_CLASSES = 4

# ---------------------- Dataset ----------------------
class CellDataset(Dataset):
    def __init__(self, X, y, donors):
        self.X = X
        self.y = y
        self.donors = donors
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.donors[i]

def _get_features(adata, idx, use_scvi):
    if use_scvi:
        if "X_scVI" not in adata.obsm_keys():
            raise KeyError("obsm['X_scVI'] not found; remove --use-scvi or add scVI.")
        Z = adata.obsm["X_scVI"][idx, :]
        X = np.asarray(Z, dtype=np.float32)
    else:
        X = adata.X[idx, :]
        # convert to dense if sparse
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
    return X

def _cap_per_donor(indices, donors, cap, seed=0):
    if cap is None: return indices
    rng = np.random.default_rng(seed)
    by_d = {}
    for i in indices:
        d = donors[i]
        by_d.setdefault(d, []).append(i)
    kept = []
    for d, idxs in by_d.items():
        if len(idxs) > cap:
            kept.extend(list(rng.choice(idxs, size=cap, replace=False)))
        else:
            kept.extend(idxs)
    kept = np.array(kept, dtype=np.int64)
    return np.sort(kept)

def load_h5ad_as_dataset(path, label_col="ADNC", cell_type=None, use_scvi=False, cap_per_donor=None):
    print(f"[load] {path}")
    ad = sc.read_h5ad(path, backed="r")
    obs = ad.obs

    # Optional filter by Supertype inside the script
    if cell_type is not None and "Supertype" in obs.columns:
        mask_ct = obs["Supertype"].astype(str) == str(cell_type)
        idx_all = np.flatnonzero(mask_ct.values)
    else:
        idx_all = np.arange(ad.n_obs)

    # Filter valid ADNC rows
    if label_col not in obs.columns:
        raise KeyError(f"obs['{label_col}'] not found.")
    lab = obs[label_col].astype(str).values
    mask_lab = np.array([l in ADNC_SET for l in lab])
    idx_all = idx_all[mask_lab[idx_all]]

    # Optionally cap per donor
    donors_all = obs["Donor ID"].astype(str).values
    idx_final = _cap_per_donor(idx_all, donors_all, cap_per_donor)

    # Build arrays
    X = _get_features(ad, idx_final, use_scvi)
    y = np.array([ADNC_MAP[lab[i]] for i in idx_final], dtype=np.int64)
    donors = donors_all[idx_final].astype(object)

    ad.file.close()
    # Convert to tensors
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return CellDataset(X_t, y_t, donors), X.shape[1]

# ---------------------- Model ----------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1, bias=False),
            nn.Dropout(0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden2, num_classes)
        )
    def forward(self, x):
        # x: (B, D)
        return self.net(x)

# ---------------------- Train / Eval ----------------------
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    tot = 0.0
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        tot += float(loss)
    return tot / max(1, len(loader))

@torch.no_grad()
def eval_cell_level(model, loader, crit, device):
    model.eval()
    tot = 0.0; n = 0
    y_true=[]; y_pred=[]
    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits, yb)
        tot += float(loss); n += 1
        pred = logits.argmax(dim=1)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    acc = (np.array(y_true)==np.array(y_pred)).mean()*100
    f1w = f1_score(y_true, y_pred, average="weighted")*100
    return tot/max(1,n), acc, f1w

@torch.no_grad()
def eval_donor_level(model, loader, device):
    """
    Bonus: donor-level aggregation by mean probabilities.
    """
    model.eval()
    from collections import defaultdict
    pool = defaultdict(list)
    donor_label = {}
    for xb, yb, donors in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        yb   = yb.cpu().numpy()
        for d, p, y in zip(donors, probs, yb):
            pool[d].append(p)
            donor_label[d] = y
    y_true=[]; y_pred=[]
    for d, plist in pool.items():
        mp = np.mean(plist, axis=0)
        y_true.append(donor_label[d])
        y_pred.append(int(np.argmax(mp)))
    acc = (np.array(y_true)==np.array(y_pred)).mean()*100
    f1w = f1_score(y_true, y_pred, average="weighted")*100
    return acc, f1w, len(pool)

# ---------------------- Main ----------------------
def main():
    # Device
    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Data
    train_ds, in_dim_tr = load_h5ad_as_dataset(
        args.train_h5ad,
        label_col=args.label_col,
        cell_type=args.cell_type,
        use_scvi=args.use_scvi,
        cap_per_donor=args.max_cells_per_donor
    )
    test_ds, in_dim_te = load_h5ad_as_dataset(
        args.test_h5ad,
        label_col=args.label_col,
        cell_type=args.cell_type,
        use_scvi=args.use_scvi,
        cap_per_donor=None  # don't cap test
    )
    assert in_dim_tr == in_dim_te, "train/test feature dim mismatch"
    in_dim = in_dim_tr
    print(f"[features] input_dim={in_dim}  classes={NUM_CLASSES}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = MLP(in_dim)
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        # Optional: data parallel over listed visible GPUs
        model = nn.DataParallel(model)
    model = model.to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    # Outdir
    os.makedirs(args.outdir, exist_ok=True)

    # Train
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, crit, device)
        te_loss, te_acc, te_f1 = eval_cell_level(model, test_loader, crit, device)
        d_acc, d_f1, n_donors = eval_donor_level(model, test_loader, device)
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} | test_cell: loss={te_loss:.4f} acc={te_acc:.2f}% f1w={te_f1:.2f}% | test_donor: acc={d_acc:.2f}% f1w={d_f1:.2f}% (N_donors={n_donors})")

        # Save
        if d_acc > best_acc:
            best_acc = d_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, f"mlp_adnc_best.pt"))
        if (epoch % args.save_every) == 0:
            torch.save(model.state_dict(), os.path.join(args.outdir, f"mlp_adnc_epoch{epoch}.pt"))

    # Export a small report
    with open(os.path.join(args.outdir, "train_config.txt"), "w") as f:
        f.write(str(vars(args)) + "\n")
        f.write(f"input_dim={in_dim}\n")

if __name__ == "__main__":
    main()
