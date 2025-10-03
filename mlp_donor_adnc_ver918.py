#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (MIL-style) for donor-level ADNC classification (4 classes).

- Donor split is given by split_donors.json (train_donors/test_donors).
- Per-cell encoder -> set pooling (mean or gated-attention) -> donor head.
- Balanced donor batch sampler: each batch has equal #donors per available ADNC class
  (and each donor contributes the same #cells via k_per_donor_train).
- Supports features from adata.X (default) or obsm["X_scVI"] (via --use-x-scvi).
- Donor-level class weights for CrossEntropyLoss.
- Early stopping based on donor_acc or donor_f1.
- Saves best model and per-epoch metrics (optional).
和上一版本不同于 按照每个ADNC分给了每个batch相同的数量
"""

import os
import re
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Args
# -----------------------
def get_args():
    ap = argparse.ArgumentParser("Donor-level MIL-MLP for ADNC (v916)")
    ap.add_argument("--data-path", required=True, help=".h5ad path")
    ap.add_argument("--split-json", required=True, help="split_donors.json")
    ap.add_argument("--donor-col", default="Donor ID")
    ap.add_argument("--adnc-col", default="ADNC")

    # optional cell filtering (e.g., Class/Subclass/Supertype)
    ap.add_argument("--filter-col", default=None, help="obs column to filter on")
    ap.add_argument("--filter-value", default=None, help="value to match")
    ap.add_argument("--icontains", action="store_true", help="case-insensitive substring match for filter")

    # features
    ap.add_argument("--use-x-scvi", action="store_true", help="use obsm['X_scVI'] as features (else adata.X)")
    ap.add_argument("--h-enc1", type=int, default=256)
    ap.add_argument("--h-enc2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--pool", choices=["mean", "attn"], default="mean")
    ap.add_argument("--head-hid", type=int, default=128)

    # training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-donors", type=int, default=12, help="donors per batch (balanced across classes)")
    ap.add_argument("--k-per-donor-train", type=int, default=2048, help="cells sampled per donor per epoch (train)")
    ap.add_argument("--k-per-donor-eval", type=int, default=6000, help="cells per donor at eval (None=all)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--early-metric", choices=["donor_acc", "donor_f1"], default="donor_acc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--steps-per-epoch", type=int, default=200)

    # outputs
    ap.add_argument("--out-dir", default="mil_out_v916")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-metrics", action="store_true")

    return ap.parse_args()


ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# -----------------------
# Data containers
# -----------------------
class DonorBags:
    """
    Hold cell indices per donor and donor labels (majority of cell labels).
    Provide per-donor subsampling for train/eval.
    """
    def __init__(self, X, y_cell, donors_cell, donor_list, k_per_donor=None, rng=None):
        self.X = X
        self.y_cell = np.asarray(y_cell, dtype=np.int64)
        self.donors_cell = np.asarray(donors_cell)
        self.donor_ids = sorted(list(set(donor_list)))
        self.k = k_per_donor
        self.rng = np.random.default_rng() if rng is None else rng

        # indices by donor
        self.idxs_by_donor = defaultdict(list)
        for i, d in enumerate(self.donors_cell):
            self.idxs_by_donor[d].append(i)

        # donor label = majority of its cells
        maj = {}
        for d in self.donor_ids:
            idxs = self.idxs_by_donor[d]
            ys = [self.y_cell[i] for i in idxs]
            vals, cnts = np.unique(ys, return_counts=True)
            maj[d] = int(vals[np.argmax(cnts)])
        self.y_donor = maj

    def subsample_indices(self, idxs, k):
        if (k is None) or (len(idxs) <= k):
            return idxs
        return self.rng.choice(idxs, size=k, replace=False).tolist()

    def get_bag(self, donor_id, k=None):
        idxs = self.idxs_by_donor[donor_id]
        use = self.subsample_indices(idxs, k)
        return use, self.y_donor[donor_id]


def collate_bags(donor_list, bags: DonorBags, X, k_cells):
    """
    Build one batch from a list of donor IDs:
    - Concatenate all their sampled cell features
    - bag_ptr marks donor segment boundaries
    - y_donor is donor-level label per bag
    """
    feats, bag_ptr, y_donors = [], [0], []
    total = 0
    for d in donor_list:
        idxs, y_d = bags.get_bag(d, k_cells)
        Xi = X[idxs]
        if sp.issparse(Xi):
            Xi = Xi.toarray()
        Xi = np.asarray(Xi, np.float32)
        feats.append(torch.from_numpy(Xi))
        total += Xi.shape[0]
        bag_ptr.append(total)
        y_donors.append(y_d)

    feats = torch.vstack(feats) if len(feats) else torch.empty(0)
    bag_ptr = torch.tensor(bag_ptr, dtype=torch.long)
    y_donors = torch.tensor(y_donors, dtype=torch.long)
    return feats, bag_ptr, y_donors


# -----------------------
# Model
# -----------------------
class CellEncoder(nn.Module):
    def __init__(self, in_dim, h1=256, h2=128, p=0.15):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p)
        )
    def forward(self, x):
        return self.enc(x)  # (Ncells, h2)


class MeanPool(nn.Module):
    def forward(self, H, bag_ptr):
        B = bag_ptr.numel() - 1
        Z = []
        for b in range(B):
            s, e = bag_ptr[b].item(), bag_ptr[b+1].item()
            Z.append(H[s:e].mean(dim=0, keepdim=True))
        return torch.vstack(Z) if Z else torch.empty(0, H.size(1), device=H.device)


class GatedAttentionPool(nn.Module):
    """ Ilse et al., 2018. """
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, H, bag_ptr):
        B = bag_ptr.numel() - 1
        Z = []
        for b in range(B):
            s, e = bag_ptr[b].item(), bag_ptr[b+1].item()
            Hb = H[s:e]
            A = torch.tanh(self.V(Hb)) * torch.sigmoid(self.U(Hb))
            a = torch.softmax(self.w(A).squeeze(-1), dim=0)
            zb = (a.unsqueeze(0) @ Hb).squeeze(0)
            Z.append(zb.unsqueeze(0))
        return torch.vstack(Z) if Z else torch.empty(0, H.size(1), device=H.device)


class DonorHead(nn.Module):
    def __init__(self, in_dim, hid=128, num_classes=4, p=0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, num_classes)
        )
    def forward(self, z):
        return self.mlp(z)


class MILNet(nn.Module):
    def __init__(self, in_dim, enc_h1=256, enc_h2=128, pool="mean", head_hid=128, p=0.15):
        super().__init__()
        self.enc = CellEncoder(in_dim, enc_h1, enc_h2, p)
        self.pool = MeanPool() if pool == "mean" else GatedAttentionPool(enc_h2)
        self.head = DonorHead(enc_h2, head_hid, 4, p)
    def forward(self, x_cells, bag_ptr):
        H = self.enc(x_cells)
        Z = self.pool(H, bag_ptr)
        logits = self.head(Z)
        return logits


# -----------------------
# Sampler: balanced donors per class
# -----------------------
class BalancedDonorBatchSampler(torch.utils.data.IterableDataset):
    """
    每个 epoch 产生 steps_per_epoch 个 batch；
    每个 batch 在“可用的 ADNC 类”上，按 per_class= batch_donors // n_classes
    进行**有放回**随机抽样，从而保证小类也能持续出现在每个 batch。
    """
    def __init__(self, donors_by_class, batch_donors=12, seed=42, classes=(0,1,2,3), steps_per_epoch=200):
        super().__init__()
        self.donors_by_class = {int(c): list(v) for c, v in donors_by_class.items()}
        self.classes = [c for c in classes if len(self.donors_by_class.get(c, [])) > 0]
        if len(self.classes) == 0:
            raise ValueError("No classes available in donors_by_class.")
        self.per_class = max(1, batch_donors // len(self.classes))
        if batch_donors % len(self.classes) != 0:
            print(f"[warn] batch_donors={batch_donors} not divisible by #classes={len(self.classes)}; "
                  f"using per_class={self.per_class}, effective batch_donors={self.per_class*len(self.classes)}")
        self.batch_donors = self.per_class * len(self.classes)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            for c in self.classes:
                pool = self.donors_by_class[c]
                # 有放回抽样，保证小类也能反复出现
                pick = self.rng.choice(pool, size=self.per_class, replace=True).tolist()
                batch.extend(pick)
            self.rng.shuffle(batch)
            yield batch


# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def evaluate(model, donors_iterable, bags, X, device, k_eval):
    model.eval()
    y_true, y_pred = [], []
    for donor_batch in donors_iterable:
        feats, bag_ptr, y_d = collate_bags(donor_batch, bags, X, k_eval)
        if feats.numel() == 0:
            continue
        feats = feats.to(device, non_blocking=True)
        bag_ptr = bag_ptr.to(device)
        logits = model(feats, bag_ptr)
        pred = logits.argmax(1).cpu()
        y_true.extend(y_d.tolist())
        y_pred.extend(pred.tolist())
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1  = f1_score(y_true, y_pred, average="weighted") * 100.0
    return acc, f1


# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- split ----
    with open(args.split_json, "r") as f:
        split = json.load(f)
    train_donors = set(split.get("train_donors") or split.get("train") or [])
    test_donors  = set(split.get("test_donors")  or split.get("test")  or [])
    if not train_donors or not test_donors:
        raise ValueError("split_donors.json must contain 'train_donors' and 'test_donors'.")

    # ---- load h5ad ----
    ad = sc.read_h5ad(args.data_path)

    # ADNC column fallback
    if args.adnc_col not in ad.obs.columns:
        if "Overall AD neuropathological Change" in ad.obs.columns:
            args.adnc_col = "Overall AD neuropathological Change"
        else:
            raise KeyError("ADNC column not found in obs.")

    # keep valid ADNC
    ad = ad[ad.obs[args.adnc_col].astype(str).isin(ADNC_ALLOWED), :].copy()

    # optional filter on obs column
    if args.filter_col and args.filter_col in ad.obs.columns and args.filter_value:
        s = ad.obs[args.filter_col].astype(str)
        if args.icontains:
            mask = s.str.contains(args.filter_value, case=False, regex=False)
        else:
            mask = (s == args.filter_value)
        ad = ad[mask, :].copy()

    donors = ad.obs[args.donor_col].astype(str).values
    y = np.array([ADNC_MAP[s] for s in ad.obs[args.adnc_col].astype(str).values], dtype=np.int64)

    # features
    if args.use_x_scvi:
        if "X_scVI" not in ad.obsm_keys():
            raise KeyError("obsm['X_scVI'] not found; drop --use-x-scvi or provide embedding.")
        X = ad.obsm["X_scVI"]
        in_dim = X.shape[1]
    else:
        X = ad.X
        in_dim = ad.n_vars

    # donors present in this file
    donors_in_file = set(np.unique(donors))
    train_d = sorted(list(donors_in_file & train_donors))
    test_d  = sorted(list(donors_in_file & test_donors))
    print(f"[donors in file] train={len(train_d)} test={len(test_d)}")

    # build donor bags (train/eval)
    bags_tr = DonorBags(X, y, donors, donor_list=train_d,
                        k_per_donor=args.k_per_donor_train,
                        rng=np.random.default_rng(args.seed))
    bags_te = DonorBags(X, y, donors, donor_list=test_d,
                        k_per_donor=args.k_per_donor_eval,
                        rng=np.random.default_rng(args.seed + 1))

    # ---- print donor/cell distribution per class (train & test) ----
    def print_dist(bags, name):
        d2y = bags.y_donor
        donors_by_cls = defaultdict(list)
        for d, yy in d2y.items(): donors_by_cls[yy].append(d)
        cells_by_cls = Counter()
        for d, idxs in bags.idxs_by_donor.items():
            if d in d2y:
                cells_by_cls[d2y[d]] += len(idxs)
        print(f"[{name}] donors per class:", {k: len(v) for k, v in sorted(donors_by_cls.items())})
        print(f"[{name}] cells  per class:", dict(sorted(cells_by_cls.items())))
    print_dist(bags_tr, "train")
    print_dist(bags_te, "test")

    # ---- donors_by_class for balanced sampler ----
    donor2y = bags_tr.y_donor
    donors_by_class = defaultdict(list)
    for d, yy in donor2y.items():
        donors_by_class[int(yy)].append(d)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MILNet(in_dim, enc_h1=args.h_enc1, enc_h2=args.h_enc2,
                   pool=args.pool, head_hid=args.head_hid, p=args.dropout).to(device)

    # ---- donor-level class weights (each donor counts once) ----
    cnt = Counter(donor2y.values())
    hist = np.array([cnt.get(c, 0) for c in range(4)], dtype=float)
    w = hist.sum() / (hist + 1e-9)
    w = w / w.sum()
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- balanced donor sampler for training ----
    sampler = BalancedDonorBatchSampler(donors_by_class, batch_donors=args.batch_donors, seed=args.seed, steps_per_epoch=args.steps_per_epoch)

    def train_epoch():
        model.train()
        run_loss, n_batches = 0.0, 0
        for donor_batch in sampler:
            feats, bag_ptr, y_d = collate_bags(donor_batch, bags_tr, X, args.k_per_donor_train)
            if feats.numel() == 0:  # safety
                continue
            feats = feats.to(device, non_blocking=True)
            bag_ptr = bag_ptr.to(device)
            y_d = y_d.to(device)

            opt.zero_grad()
            logits = model(feats, bag_ptr)
            loss = ce(logits, y_d)
            loss.backward()
            opt.step()

            run_loss += loss.item()
            n_batches += 1
        return run_loss / max(1, n_batches)

    # ---- evaluation donor iterator (fixed order batches of donors) ----
    def eval_iterable(donor_ids, batch_donors):
        # simple fixed-size chunks (not necessarily balanced)
        B = len(donor_ids)
        for i in range(0, B, batch_donors):
            yield donor_ids[i:i+batch_donors]

    best, best_ep, patience = -np.inf, -1, args.patience
    best_state = None
    metrics_hist = []

    # fixed donor lists for eval
    te_chunks = list(eval_iterable(test_d, args.batch_donors))

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch()
        da, df = evaluate(model, te_chunks, bags_te, X, device, args.k_per_donor_eval)
        print(f"[epoch {ep}] train_loss={tr_loss:.4f} | DONOR acc={da:.2f}% f1={df:.2f}%")
        metrics_hist.append({"epoch": ep, "train_loss": tr_loss, "donor_acc": da, "donor_f1": df})

        score = da if args.early_metric == "donor_acc" else df
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[early stop] best {args.early_metric}={best:.2f}% @epoch {best_ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        da, df = evaluate(model, te_chunks, bags_te, X, device, args.k_per_donor_eval)
        print(f"[best @epoch {best_ep}] DONOR acc={da:.2f}% f1={df:.2f}%")

    # ---- save outputs ----
    if args.save_metrics:
        pd.DataFrame(metrics_hist).to_csv(os.path.join(args.out_dir, "metrics_by_epoch.csv"), index=False)
        print(f"[save] metrics -> {os.path.join(args.out_dir, 'metrics_by_epoch.csv')}")
    if args.save_model and best_state is not None:
        torch.save(best_state, os.path.join(args.out_dir, "best_mlp_mil_v916.pth"))
        print(f"[save] model -> {os.path.join(args.out_dir, 'best_mlp_mil_v916.pth')}")


if __name__ == "__main__":
    main()
