#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIL-MLP for ADNC (4-class):
- Per-cell encoder -> set pooling (mean or gated-attention) -> donor head
- Donor-level 80/20 split is fixed via split_donors.json
- Each epoch subsample up to K cells per donor to control memory/time
通过细胞预测ADNC stage从cell lv学习
"""

import os, json, argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------- Args -------------
def get_args():
    ap = argparse.ArgumentParser("MIL MLP ADNC")
    ap.add_argument("--data-path", required=True, help=".h5ad path")
    ap.add_argument("--split-json", required=True, help="split_donors.json")
    ap.add_argument("--donor-col", default="Donor ID")
    ap.add_argument("--adnc-col", default="ADNC")
    ap.add_argument("--filter-col", default=None, help="optional: Class/Subclass/Supertype")
    ap.add_argument("--filter-value", default=None, help="optional: e.g. 'Astrocyte' or regex (if --icontains)")
    ap.add_argument("--icontains", action="store_true", help="case-insensitive substring match for filter")
    ap.add_argument("--use-x-scvi", action="store_true", help="use obsm['X_scVI'] as features")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-donors", type=int, default=16, help="how many donors per batch")
    ap.add_argument("--k-per-donor-train", type=int, default=2048, help="cells sampled per donor each epoch (train)")
    ap.add_argument("--k-per-donor-eval", type=int, default=8192, help="cells sampled per donor for eval (None=all)")
    ap.add_argument("--h-enc1", type=int, default=256)
    ap.add_argument("--h-enc2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--pool", choices=["mean", "attn"], default="attn")
    ap.add_argument("--head-hid", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--early-metric", choices=["donor_acc", "donor_f1"], default="donor_acc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="mil_out")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-metrics", action="store_true")
    return ap.parse_args()

ADNC_MAP = {'Not AD':0, 'Low':1, 'Intermediate':2, 'High':3}
ADNC_ALLOWED = set(ADNC_MAP.keys())

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ------------- Data -------------
class DonorBags:
    """
    Store cell features & labels per donor.
    Provide subsampled bags for train/eval.
    """
    def __init__(self, X, y, donors, donor_list, k_per_donor=None, rng=None):
        self.X = X
        self.y_cell = y      # cell-level numeric labels (all from their donor)
        self.donors_cell = donors
        self.donor_ids = sorted(list(set(donor_list)))
        self.k = k_per_donor
        self.rng = np.random.default_rng() if rng is None else rng

        # map donor -> cell indices
        self.idxs_by_donor = {}
        for i, d in enumerate(self.donors_cell):
            self.idxs_by_donor.setdefault(d, []).append(i)

        # donor-level labels (use donor's majority)
        maj = {}
        for d, idxs in self.idxs_by_donor.items():
            if d in self.donor_ids:
                ys = [self.y_cell[i] for i in idxs]
                # majority
                vals, cnts = np.unique(ys, return_counts=True)
                maj[d] = int(vals[np.argmax(cnts)])
        self.y_donor = maj

    def __len__(self):
        return len(self.donor_ids)

    def subsample_indices(self, idxs, k):
        if (k is None) or (len(idxs) <= k):
            return idxs
        return self.rng.choice(idxs, size=k, replace=False).tolist()

    def get_bag(self, d, k=None):
        idxs = self.idxs_by_donor[d]
        use = self.subsample_indices(idxs, k)
        return use, self.y_donor[d]

class DonorBatchDataset(Dataset):
    """
    Iterable over donors; collate_fn packs variable-sized bags into one batch.
    """
    def __init__(self, bags: DonorBags, donors):
        self.bags = bags
        self.donors = list(donors)

    def __len__(self): return len(self.donors)

    def __getitem__(self, i):
        d = self.donors[i]
        return d  # real data assembled in collate_fn

def collate_bags(donor_list, bags: DonorBags, X, k_cells):
    # build one batch (concat all cells; keep offsets)
    feats, labels, bag_ptr, y_donors = [], [], [0], []
    total = 0
    for d in donor_list:
        idxs, y_d = bags.get_bag(d, k_cells)
        # fetch rows
        Xi = X[idxs]
        if sp.issparse(Xi):
            Xi = Xi.toarray()
        Xi = np.asarray(Xi, np.float32)
        feats.append(torch.from_numpy(Xi))
        total += Xi.shape[0]
        bag_ptr.append(total)       # cumulative
        y_donors.append(y_d)
    feats = torch.vstack(feats) if feats else torch.empty(0)
    y_donors = torch.tensor(y_donors, dtype=torch.long)
    bag_ptr = torch.tensor(bag_ptr, dtype=torch.long)  # len = batch_donors+1
    return feats, bag_ptr, y_donors

# ------------- Model -------------
class CellEncoder(nn.Module):
    def __init__(self, in_dim, h1=256, h2=128, p=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p)
        )
    def forward(self, x):  # (Ncells, in_dim) -> (Ncells, h2)
        return self.enc(x)

class MeanPool(nn.Module):
    def forward(self, H, bag_ptr):
        # H: (Ncells, D); bag_ptr: [0, n1, n1+n2, ...]
        B = bag_ptr.numel() - 1
        Z = []
        for b in range(B):
            s, e = bag_ptr[b].item(), bag_ptr[b+1].item()
            Z.append(H[s:e].mean(dim=0, keepdim=True))
        return torch.vstack(Z) if Z else torch.empty(0, H.size(1), device=H.device)

class GatedAttentionPool(nn.Module):
    # Ilse et al., 2018
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
            Hb = H[s:e]  # (n_b, D)
            A = torch.tanh(self.V(Hb)) * torch.sigmoid(self.U(Hb))  # (n_b, A)
            a = torch.softmax(self.w(A).squeeze(-1), dim=0)         # (n_b,)
            zb = (a.unsqueeze(0) @ Hb).squeeze(0)                   # (D,)
            Z.append(zb.unsqueeze(0))
        return torch.vstack(Z) if Z else torch.empty(0, H.size(1), device=H.device)

class DonorHead(nn.Module):
    def __init__(self, in_dim, hid=128, num_classes=4, p=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, num_classes)
        )
    def forward(self, z): return self.mlp(z)

class MILNet(nn.Module):
    def __init__(self, in_dim, enc_h1=256, enc_h2=128, pool="attn", head_hid=128, p=0.2):
        super().__init__()
        self.enc = CellEncoder(in_dim, enc_h1, enc_h2, p)
        self.pool = GatedAttentionPool(enc_h2) if pool=="attn" else MeanPool()
        self.head = DonorHead(enc_h2, head_hid, 4, p)
    def forward(self, x_cells, bag_ptr):
        H = self.enc(x_cells)              # (Ncells, enc_h2)
        Z = self.pool(H, bag_ptr)          # (B, enc_h2)
        logits = self.head(Z)              # (B, 4)
        return logits

class BalancedDonorBatchSampler(torch.utils.data.IterableDataset):
    """
    每次 __iter__ 产出一批 donor ID，保证四个 ADNC 类 donor 数量相同。
    """
    def __init__(self, donors_by_class, batch_donors=16, seed=42):
        super().__init__()
        self.donors_by_class = {c: list(v) for c, v in donors_by_class.items()}
        self.batch_donors = batch_donors
        self.per_class = batch_donors // 4  # 要求能被4整除
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        # 每个 epoch: 先打乱每个类的 donor 列表
        pools = {c: self.rng.permutation(v).tolist() for c, v in self.donors_by_class.items()}
        # 取该 epoch 能组多少完整 batch（受最小类限制）
        min_len = min(len(v) for v in pools.values()) if pools else 0
        n_batches = min_len // self.per_class
        for b in range(n_batches):
            batch = []
            for c in range(4):
                start = b * self.per_class
                end   = (b + 1) * self.per_class
                batch.extend(pools[c][start:end])
            # 再整体打乱一下 batch 内 donor 顺序
            self.rng.shuffle(batch)
            yield batch

# ------------- Train/Eval -------------
@torch.no_grad()
def evaluate(model, loader, bags, X, device, k_eval):
    model.eval()
    y_true, y_pred = [], []
    for donor_batch in loader:
        feats, bag_ptr, y_d = collate_bags(donor_batch, bags, X, k_eval)
        if feats.numel() == 0: continue
        feats = feats.to(device, non_blocking=True)
        bag_ptr = bag_ptr.to(device)
        logits = model(feats, bag_ptr)       # (B,4)
        pred = logits.argmax(1).cpu()
        y_true.extend(y_d.tolist()); y_pred.extend(pred.tolist())
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)*100.0
    f1  = f1_score(y_true, y_pred, average="weighted")*100.0
    return acc, f1

def main():
    args = get_args(); set_seed(args.seed); os.makedirs(args.out_dir, exist_ok=True)

    # split
    with open(args.split_json,"r") as f: split = json.load(f)
    train_donors = split.get("train_donors") or split.get("train") or []
    test_donors  = split.get("test_donors")  or split.get("test")  or []
    train_donors, test_donors = set(train_donors), set(test_donors)

    # load h5ad
    ad = sc.read_h5ad(args.data_path)
    # filter ADNC
    if args.adnc_col not in ad.obs.columns:
        if "Overall AD neuropathological Change" in ad.obs.columns:
            args.adnc_col = "Overall AD neuropathological Change"
        else:
            raise KeyError("ADNC column not found.")
    ad = ad[ad.obs[args.adnc_col].astype(str).isin(ADNC_ALLOWED), :].copy()

    # optional filter by cell type level
    if args.filter_col and args.filter_col in ad.obs.columns and args.filter_value:
        s = ad.obs[args.filter_col].astype(str)
        mask = s.str.contains(args.filter_value, case=not args.icontains, regex=False) if args.icontains \
               else (s == args.filter_value)
        ad = ad[mask, :].copy()

    donors = ad.obs[args.donor_col].astype(str).values
    y = np.array([ADNC_MAP[s] for s in ad.obs[args.adnc_col].astype(str).values], np.int64)

    # features
    if args.use_x_scvi:
        if "X_scVI" not in ad.obsm_keys(): raise KeyError("X_scVI not found.")
        X = ad.obsm["X_scVI"]
        in_dim = X.shape[1]
    else:
        X = ad.X
        in_dim = ad.n_vars

    # split to donor sets present in this file
    donors_in_file = set(np.unique(donors))
    train_d = sorted(list(donors_in_file & train_donors))
    test_d  = sorted(list(donors_in_file & test_donors))
    print(f"[donors in file] train={len(train_d)} test={len(test_d)}")

    # build donor bags
    bags_tr = DonorBags(X, y, donors, donor_list=train_d, k_per_donor=args.k_per_donor_train,
                        rng=np.random.default_rng(args.seed))
    bags_te = DonorBags(X, y, donors, donor_list=test_d,  k_per_donor=args.k_per_donor_eval,
                        rng=np.random.default_rng(args.seed+1))

    # donor datasets: they yield donor IDs; collate builds variable-size bags
    te_ds = [d for d in test_d]
    te_loader = DataLoader(te_ds, batch_size=args.batch_donors, shuffle=False)

    # 要求 batch_donors 能被 4 整除
    assert args.batch_donors % 4 == 0, "batch-donors 必须能被4整除以便四类平衡"

    sampler = BalancedDonorBatchSampler(donors_by_class, batch_donors=args.batch_donors, seed=args.seed)

    def donor_loader_epoch():
        # sampler 每个 epoch 产出若干“donor 批次”，我们在循环里逐批训练
        for donor_batch in sampler:
            yield donor_batch

    # model/opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MILNet(in_dim, enc_h1=args.h_enc1, enc_h2=args.h_enc2,
                   pool=args.pool, head_hid=args.head_hid, p=args.dropout).to(device)
    # ==== donor-level class weights (each donor counts once) ====
    from collections import Counter
    donor2y = bags_tr.y_donor  # dict: donor -> label (0..3)
    cnt = Counter(donor2y.values())
    hist = np.array([cnt.get(c, 0) for c in range(4)], dtype=float)
    w = hist.sum() / (hist + 1e-9)      # inverse frequency
    w = w / w.sum()
    class_weights = torch.tensor(w, dtype=torch.float32).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # === 统计每个 ADNC 类的 donor 数、cell 数（训练集）
    from collections import Counter, defaultdict
    donor2y = bags_tr.y_donor  # dict: donor -> label (0..3)

    donors_by_class = defaultdict(list)
    for d, y in donor2y.items():
        donors_by_class[y].append(d)

    cells_by_class = Counter()
    for d, idxs in bags_tr.idxs_by_donor.items():
        y = donor2y[d]
        cells_by_class[y] += len(idxs)

    print("[train] donors per class:", {k: len(v) for k, v in sorted(donors_by_class.items())})
    print("[train] cells  per class:", dict(sorted(cells_by_class.items())))


    best, best_ep, patience = -np.inf, -1, args.patience
    hist = []
    # 训练循环
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n_batches = 0.0, 0
        for donor_batch in donor_loader_epoch():
            feats, bag_ptr, y_d = collate_bags(donor_batch, bags_tr, X, args.k_per_donor_train)
            if feats.numel() == 0: 
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
        tr_loss = run_loss / max(1, n_batches)

        # eval on donors
        da, df = evaluate(model, te_loader, bags_te, X, device, args.k_per_donor_eval)
        row = {"epoch": ep, "train_loss": tr_loss, "donor_acc": da, "donor_f1": df}
        hist.append(row)
        print(f"[epoch {ep}] train_loss={tr_loss:.4f} | DONOR acc={da:.2f}% f1={df:.2f}%")

        score = da if args.early_metric=="donor_acc" else df
        if score > best:
            best, best_ep = score, ep
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"[early stop] best {args.early_metric}={best:.2f}% @epoch {best_ep}")
                break

    # restore best and final eval
    model.load_state_dict(best_state)
    da, df = evaluate(model, te_loader, bags_te, X, device, args.k_per_donor_eval)
    print(f"[best @epoch {best_ep}] DONOR acc={da:.2f}% f1={df:.2f}%")

    # save outputs
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.save_metrics:
        pd.DataFrame(hist).to_csv(os.path.join(out_dir, "metrics_by_epoch.csv"), index=False)
    if args.save_model:
        torch.save(model.state_dict(), os.path.join(out_dir, "best_mil_mlp.pth"))

if __name__ == "__main__":
    main()
