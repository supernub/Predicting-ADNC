#!/usr/bin/env python3
"""
# 先固定 split（一次即可）
python make_donor_split.py \
  --h5ads ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
          ../training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad \
  --out split_donors.json --test-size 0.2 --seed 42

# 然后训练某个 cell type 的 MLP（一直用同一 split）
python mlp_adnc_per_celltype_split.py \
  --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
  --split-json split_donors.json \
  --cell-type "Astrocyte" \
  --celltype-col "Supertype" \
  --adnc-col "ADNC" \
  --donor-col "Donor ID" \
  --use-x-scvi \
  --epochs 10 --batch-size 256

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a cell-level MLP (ADNC 4-class) for a specific cell type, using a fixed
donor-level 8:2 split loaded from split_donors.json. At evaluation, aggregate
cell probabilities per (donor, cell type) by mean to produce donor×4 features.

Author: your teammate :)
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


# -----------------------
# Argparse
# -----------------------
def get_args():
    ap = argparse.ArgumentParser(description="Cell-type MLP for ADNC (4-class) with fixed donor split")

    ap.add_argument("--data-path", dest="data_path", required=True, help="input .h5ad path (A9/MTG)")
    ap.add_argument("--split-json", dest="split_json", required=True, help="split_donors.json path")
    ap.add_argument("--cell-type", dest="cell_type", required=True, help="e.g. 'Astrocyte'")

    ap.add_argument("--celltype-col", dest="celltype_col", default="Supertype",
                    help="column in obs for cell type (e.g. 'Supertype' or 'Subclass')")
    ap.add_argument("--adnc-col", dest="adnc_col", default="ADNC",
                    help="ADNC column name in obs (e.g. 'ADNC' or 'Overall AD neuropathological Change')")
    ap.add_argument("--donor-col", dest="donor_col", default="Donor ID",
                    help="donor column name in obs")

    ap.add_argument("--use-x-scvi", dest="use_x_scvi", action="store_true",
                    help="use obsm['X_scVI'] as features (recommended)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", dest="out_dir", default="donor_features",
                    help="output directory to save donor×4 probabilities and (optionally) model")

    ap.add_argument("--save-model", dest="save_model", action="store_true",
                    help="save final model weights to out_dir/models/")
    return ap.parse_args()


# -----------------------
# Utils
# -----------------------
ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_adnc_col(obs, user_key):
    """Robustly find the ADNC column."""
    for k in [user_key, "ADNC", "Overall AD neuropathological Change"]:
        if k in obs.columns:
            return k
    raise KeyError(f"Cannot find ADNC column. Checked: {user_key}, 'ADNC', 'Overall AD neuropathological Change'.")


class CellDataset(Dataset):
    """Memory-friendly per-cell dataset reading sparse rows on the fly."""
    def __init__(self, X, y, donors):
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.donors = np.asarray(donors)
        self.is_sparse = sp.issparse(X)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        xi = self.X[i]
        if self.is_sparse:
            xi = xi.toarray().ravel()
        else:
            xi = np.asarray(xi).ravel()
        return torch.from_numpy(xi.astype(np.float32)), torch.tensor(self.y[i]), self.donors[i]


class MLP(nn.Module):
    def __init__(self, in_dim, h1=256, h2=256, num_classes=4, p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1, bias=False),
            nn.Dropout(p), nn.ReLU(),
            nn.Linear(h1, h2),
            nn.Dropout(p), nn.ReLU(),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate_and_aggregate(model, loader, device):
    """Return cell-level metrics and donor-level aggregated prob features."""
    model.eval()
    y_true, y_pred = [], []
    buckets = defaultdict(list)  # donor -> list of prob vectors
    donor_label = {}             # donor -> numeric label

    for xb, yb, donors_b in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())
        for d, p, y_ in zip(donors_b, probs, yb.numpy()):
            buckets[d].append(p)
            donor_label[d] = int(y_)

    cell_acc = accuracy_score(y_true, y_pred) * 100.0
    cell_f1 = f1_score(y_true, y_pred, average="weighted") * 100.0

    donor_feat = {}
    d_true, d_pred = [], []
    for d, plist in buckets.items():
        mp = np.mean(np.stack(plist, axis=0), axis=0)  # (4,)
        donor_feat[d] = mp
        d_true.append(donor_label[d])
        d_pred.append(int(mp.argmax()))
    donor_acc = accuracy_score(d_true, d_pred) * 100.0
    donor_f1 = f1_score(d_true, d_pred, average="weighted") * 100.0

    return {
        "cell_acc": cell_acc,
        "cell_f1": cell_f1,
        "donor_acc": donor_acc,
        "donor_f1": donor_f1,
        "donor_feat": donor_feat
    }


# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] reading split: {args.split_json}")
    with open(args.split_json, "r") as f:
        split = json.load(f)
    train_donors = set(split.get("train_donors") or split.get("train") or [])
    test_donors = set(split.get("test_donors") or split.get("test") or [])
    if not train_donors or not test_donors:
        raise ValueError("split_donors.json must contain 'train_donors' and 'test_donors' (or 'train'/'test').")

    print(f"[info] reading h5ad: {args.data_path}")
    ad = sc.read_h5ad(args.data_path)  # load into memory after filtering by cell type
    ct_col = args.celltype_col
    dn_col = args.donor_col
    adnc_col = pick_adnc_col(ad.obs, args.adnc_col)

    if ct_col not in ad.obs.columns:
        raise KeyError(f"'{ct_col}' not in obs. First few columns: {list(ad.obs.columns)[:20]}")

    # filter by cell type and valid ADNC
    mask_ct = ad.obs[ct_col].astype(str) == args.cell_type
    mask_y = ad.obs[adnc_col].astype(str).isin(ADNC_ALLOWED)
    ad = ad[mask_ct & mask_y, :].copy()
    if ad.n_obs == 0:
        raise ValueError(f"No cells for cell_type='{args.cell_type}' with valid ADNC in {args.data_path}")

    donors = ad.obs[dn_col].astype(str).values
    y_num = np.array([ADNC_MAP[s] for s in ad.obs[adnc_col].astype(str).values], dtype=np.int64)

    # features
    if args.use_x_scvi:
        if "X_scVI" not in ad.obsm_keys():
            raise KeyError("obsm['X_scVI'] not found; drop --use-x-scvi or provide scVI embedding.")
        X = ad.obsm["X_scVI"]
        in_dim = X.shape[1]
    else:
        X = ad.X  # (n_cells, 36601), may be sparse
        in_dim = ad.n_vars

    # donor split on cells
    tr_mask = np.isin(donors, list(train_donors))
    te_mask = np.isin(donors, list(test_donors))

    X_tr, y_tr, d_tr = X[tr_mask], y_num[tr_mask], donors[tr_mask]
    X_te, y_te, d_te = X[te_mask], y_num[te_mask], donors[te_mask]

    print(f"[split] train cells: {len(y_tr)} | test cells: {len(y_te)} | "
          f"train donors: {len(train_donors)} | test donors: {len(test_donors)}")

    # dataloaders
    train_loader = DataLoader(
        CellDataset(X_tr, y_tr, d_tr),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = DataLoader(
        CellDataset(X_te, y_te, d_te),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=in_dim, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # train/eval
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += float(loss)
        print(f"[epoch {ep}] train_loss={run_loss/len(train_loader):.4f}")

        metrics = evaluate_and_aggregate(model, test_loader, device)
        print(f"[epoch {ep}] cell acc={metrics['cell_acc']:.2f}%  f1={metrics['cell_f1']:.2f}%  |  "
              f"DONOR acc={metrics['donor_acc']:.2f}%  f1={metrics['donor_f1']:.2f}%")

    # save donor×4 probs (for this cell type)
    feat_dir = os.path.join(args.out_dir, args.cell_type)
    os.makedirs(feat_dir, exist_ok=True)
    cols = [f"ADNC_prob_{k}" for k in range(4)]
    df = pd.DataFrame.from_dict(metrics["donor_feat"], orient="index", columns=cols)
    df.index.name = "Donor"
    out_csv = os.path.join(feat_dir, f"{args.cell_type}.donor_ADNC_probs.csv")
    df.to_csv(out_csv)
    print(f"[save] donor features -> {out_csv}")

    # optionally save model
    if args.save_model:
        mdir = os.path.join(args.out_dir, "models")
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, f"mlp_{args.cell_type}_adnc.pth")
        torch.save(model.state_dict(), mpath)
        print(f"[save] model -> {mpath}")


if __name__ == "__main__":
    main()
