#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-level ADNC prediction with Transformer (gene-as-token) + CORAL (ordinal)
- Split at cell level via JSON: {"train_indices":[...], "test_indices":[...], "adnc_col": "..."}
- Tokenization: per-cell Top-K nonzero genes -> (gene_id token, expression value)
- Embedding: gene embedding + FiLM(expr) modulation
- Encoder: Transformer encoder-only, CLS pooling
- Head: CORAL (3 logits for y>k, k=0..2) to respect ADNC order
- Training: cell-level loss/metrics (Acc, F1, QWK), AMP optional
- Parallel: optional DataParallel via --device-ids

Save best weights (no "module." prefix) & epoch metrics if requested.

python transformer_ADNC_cell_v925.py \
  --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
  --split-json cell_split_A9_80_20.json \
  --adnc-col "ADNC" \
  --top-k 512 --d-model 256 --layers 4 --heads 8 --ffn 512 \
  --batch-size 256 --epochs 10 --lr 3e-4 --weight-decay 1e-4 \
  --amp --out-dir runs_cell_tx_coral_a9 --save-model --save-metrics

"""

import os, json, argparse, random
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

# -----------------------
# ADNC mapping
# -----------------------
ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def row_topk_tokens(row, k=512, log1p=False):
    """
    From one cell row (1 x n_genes), select top-k nonzero genes by expression.
    Return: gene_ids (np.int64[L]), expr_vals (np.float32[L])
    """
    if sp.issparse(row):
        row = row.tocsr()
        data = row.data
        cols = row.indices
    else:
        arr = np.asarray(row).ravel()
        cols = np.where(arr > 0)[0]
        data = arr[cols]

    if len(cols) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    kk = min(k, len(cols))
    sel = np.argpartition(-data, kth=kk-1)[:kk]
    order = np.argsort(-data[sel])
    sel = sel[order]
    gene_ids = cols[sel].astype(np.int64)
    expr_vals = data[sel].astype(np.float32)
    if log1p:
        expr_vals = np.log1p(expr_vals)
    return gene_ids, expr_vals

def pad_collate(batch, pad_token_id=-1, pad_expr=0.0):
    """
    batch: list of (gene_ids, expr_vals, label_int)
    Output tensors:
      gene_ids:  (B, Lmax) Long
      expr_vals: (B, Lmax) Float
      attn_mask: (B, Lmax) Bool  (True = mask/pad)
      labels:    (B,)     Long   in [0..3]
    """
    gene_seqs, expr_seqs, labels = zip(*batch)
    L = max((len(g) for g in gene_seqs), default=1)
    B = len(batch)
    gene_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    expr_vals = torch.full((B, L), pad_expr, dtype=torch.float32)
    attn_mask = torch.ones((B, L), dtype=torch.bool)
    for i, (g, e) in enumerate(zip(gene_seqs, expr_seqs)):
        if len(g) == 0:
            continue
        gene_ids[i, :len(g)] = torch.from_numpy(g)
        expr_vals[i, :len(e)] = torch.from_numpy(e)
        attn_mask[i, :len(g)] = False
    labels = torch.tensor(labels, dtype=torch.long)
    return gene_ids, expr_vals, attn_mask, labels

# -----------------------
# Dataset (cell-level, backed='r', lazy-open per worker)
# -----------------------
class CellTokenDataset(Dataset):
    """
    关键点：
    - 不在 __init__ 里打开 h5ad（self.ad = None）
    - __getitem__ 里按需在当前 worker 进程里 lazy open（self._ensure_open()）
    - 通过 __getstate__/__setstate__ 确保 DataLoader 进程间传递时不携带句柄
    - labels_all / n_vars 在主进程一次性读出后缓存（不保留 ad 句柄）
    """
    def __init__(self, h5ad_path, indices, adnc_col="ADNC", top_k=512, log1p=False):
        self.path = h5ad_path
        self.ad = None                 # 懒打开：不要在主进程持有文件句柄
        self.idx = np.array(indices, dtype=np.int64)
        self.top_k = int(top_k)
        self.log1p = bool(log1p)

        # 仅为获取 labels 和 n_vars 做一次“短暂打开”，立刻释放（不保留句柄）
        tmp = sc.read_h5ad(h5ad_path, backed="r")
        if adnc_col not in tmp.obs_keys():
            if "Overall AD neuropathological Change" in tmp.obs_keys():
                adnc_col = "Overall AD neuropathological Change"
            else:
                raise KeyError(f"ADNC column '{adnc_col}' not found.")
        adnc = tmp.obs[adnc_col].astype(str).values
        ok = np.isin(adnc, list(ADNC_ALLOWED))
        # 过滤掉 split 中不合法的细胞索引，避免 __getitem__ 再次判断
        self.idx = self.idx[ok[self.idx]]
        self.labels_all = np.array([ADNC_MAP[s] for s in adnc], dtype=np.int64)
        self.n_vars = tmp.n_vars
        del tmp  # 立即释放临时 AnnData（关闭 h5 句柄）

    # ---- pickle 安全：不要把句柄传给子进程 ----
    def __getstate__(self):
        state = self.__dict__.copy()
        state["ad"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ad = None  # 子进程恢复时也处于未打开状态

    # ---- 每个 worker 的懒打开 ----
    def _ensure_open(self):
        if self.ad is None:
            # 每个 worker 进程里各自打开 backed='r'，句柄互不干扰
            self.ad = sc.read_h5ad(self.path, backed="r")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        self._ensure_open()
        ridx = int(self.idx[i])
        row = self.ad.X[ridx]  # 1 x n_vars
        g, e = row_topk_tokens(row, k=self.top_k, log1p=self.log1p)
        y = int(self.labels_all[ridx])
        return g, e, y

# -----------------------
# Model: Embedding + FiLM + Transformer + CORAL
# -----------------------
class FiLM(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.to_gamma = nn.Linear(1, d_model)
        self.to_beta  = nn.Linear(1, d_model)
    def forward(self, h, x):  # h:(B,L,D), x:(B,L,1)
        gamma = self.to_gamma(x)
        beta  = self.to_beta(x)
        return (1.0 + gamma) * h + beta

class GeneTokenEmbed(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.gene_emb = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.film = FiLM(d_model)
    def forward(self, gene_ids, expr_vals):
        # pad token id = -1; clamp 到 [0..vocab-1]（pad位置会被 mask）
        gid = gene_ids.clamp(min=0)
        h = self.gene_emb(gid)                 # (B,L,D)
        x = expr_vals.unsqueeze(-1)            # (B,L,1)
        h = self.film(h, x)
        return self.ln(h)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, heads=8, layers=4, ffn=512, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
    def forward(self, X, attn_mask):
        B = X.size(0)
        cls = self.cls.expand(B, -1, -1)      # (B,1,D)
        X = torch.cat([cls, X], dim=1)        # (B,1+L,D)
        pad = torch.zeros((B,1), dtype=attn_mask.dtype, device=attn_mask.device)
        m = torch.cat([pad, attn_mask], dim=1)
        H = self.enc(X, src_key_padding_mask=m)
        return H[:,0,:]                       # CLS

class CoralHead(nn.Module):
    def __init__(self, in_dim, K=3, hid=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, K)
        )
    def forward(self, z):  # (B,D)
        return self.net(z) # (B,K)

def coral_loss(logits, y, K=3):
    y = y.unsqueeze(1)  # (B,1)
    ks = torch.arange(K, device=logits.device).unsqueeze(0)  # (1,K)
    tgt = (y > ks).float()
    return F.binary_cross_entropy_with_logits(logits, tgt)

@torch.no_grad()
def coral_decode(logits):  # (B,K)
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)  # in [0..K]

class CellTransformerCoral(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=8, layers=4, ffn=512, dropout=0.1, K=3):
        super().__init__()
        self.embed = GeneTokenEmbed(vocab_size, d_model)
        self.backbone = TransformerEncoder(d_model, heads, layers, ffn, dropout)
        self.head = CoralHead(d_model, K=K, hid=128, p=dropout)
    def forward(self, gene_ids, expr_vals, attn_mask):
        X = self.embed(gene_ids, expr_vals)     # (B,L,D)
        z = self.backbone(X, attn_mask)         # (B,D)
        logits_k = self.head(z)                 # (B,K)
        return logits_k

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, dl, optimizer, device, scaler=None, amp=True, log_every=100):
    model.train()
    run_loss, n = 0.0, 0
    for step, (gids, expr, mask, y) in enumerate(dl, 1):
        gids = gids.to(device, non_blocking=True)
        expr = expr.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            with autocast(dtype=torch.float16):
                logits = model(gids, expr, mask)
                loss = coral_loss(logits, y, K=3)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(gids, expr, mask)
            loss = coral_loss(logits, y, K=3)
            loss.backward()
            optimizer.step()

        run_loss += float(loss.detach().cpu())
        n += 1
        if log_every > 0 and (step % log_every == 0):
            print(f"[train] step {step}/{len(dl)} loss={run_loss/n:.4f}")
    return run_loss / max(1, n)

@torch.no_grad()
def evaluate(model, dl, device, amp=True, log_every=100):
    model.eval()
    y_true, y_pred = [], []
    run_loss, n = 0.0, 0
    total = len(dl)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    print(f"[eval] start: {total} batches")

    for i, (gids, expr, mask, y) in enumerate(dl, 1):
        gids = gids.to(device, non_blocking=True)
        expr = expr.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        if amp:
            # 新写法，去掉 FutureWarning
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(gids, expr, mask)
                loss = coral_loss(logits, y, K=3)
        else:
            logits = model(gids, expr, mask)
            loss = coral_loss(logits, y, K=3)

        pred = coral_decode(logits)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        run_loss += float(loss.detach().cpu()); n += 1

        if log_every > 0 and (i % log_every == 0 or i == total):
            print(f"[eval] {i}/{total} done")

    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1  = f1_score(y_true, y_pred, average="weighted") * 100.0
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic") * 100.0
    return run_loss / max(1, n), acc, f1, qwk

# -----------------------
# Args & main
# -----------------------
def parse_list(arg_value):
    if isinstance(arg_value, list): return arg_value
    return [int(item) for item in str(arg_value).split(",") if str(item).strip()!=""]

def get_args():
    ap = argparse.ArgumentParser("Cell-level Transformer + CORAL for ADNC (cell split)")
    # data
    ap.add_argument("--data-path", required=True, help=".h5ad path")
    ap.add_argument("--split-json", required=True, help="JSON with train_indices/test_indices")
    ap.add_argument("--adnc-col", default="ADNC")
    ap.add_argument("--log1p", action="store_true", help="apply log1p to expression values")
    # tokenizer / model
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    # train
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-log-every", type=int, default=100)
    ap.add_argument("--amp", action="store_true")
    # parallel
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--device-ids", type=parse_list, default=[0], help="for DataParallel, e.g. 0,1,2,3")
    # output
    ap.add_argument("--out-dir", default="runs_cell_tx_coral")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-metrics", action="store_true")
    ap.add_argument("--eval-log-every", type=int, default=100)
    return ap.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load split
    with open(args.split_json, "r") as f:
        sp = json.load(f)
    # 兼容不同字段名
    tr_idx = sp.get("train_indices") or sp.get("train_cells") or sp.get("train")
    te_idx = sp.get("test_indices")  or sp.get("test_cells")  or sp.get("test")
    if tr_idx is None or te_idx is None:
        raise ValueError("split json must contain 'train_indices' and 'test_indices' (or 'train_cells'/'test_cells').")
    adnc_col = sp.get("adnc_col", args.adnc_col)

    # Build datasets/dataloaders
    ds_tr = CellTokenDataset(args.data_path, tr_idx, adnc_col=adnc_col, top_k=args.top_k, log1p=args.log1p)
    ds_te = CellTokenDataset(args.data_path, te_idx, adnc_col=adnc_col, top_k=args.top_k, log1p=args.log1p)
    vocab_size = ds_tr.n_vars  # gene_id = 列索引

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=0, pin_memory=True, collate_fn=pad_collate)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True, collate_fn=pad_collate)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Model
    model = CellTransformerCoral(
        vocab_size=vocab_size,
        d_model=args.d_model, heads=args.heads, layers=args.layers,
        ffn=args.ffn, dropout=args.dropout, K=3
    )

    # DataParallel (optional)
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if args.amp else None

    # Print class balance
    y_tr = [ds_tr.labels_all[i] for i in ds_tr.idx]
    cnt_tr = Counter(y_tr)
    print("[train] cells per ADNC:", dict(sorted(cnt_tr.items())))
    y_te = [ds_te.labels_all[i] for i in ds_te.idx]
    cnt_te = Counter(y_te)
    print("[test ] cells per ADNC:", dict(sorted(cnt_te.items())))

    # Train
    best, best_ep = -np.inf, -1
    best_state = None
    metrics = []

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, device, scaler, amp=args.amp, log_every=args.train_log_every)
        te_loss, te_acc, te_f1, te_qwk = evaluate(model, dl_te, device, amp=args.amp, log_every=args.eval_log_every)
        print(f"[epoch {ep}] train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | acc={te_acc:.2f}% f1={te_f1:.2f}% qwk={te_qwk:.2f}%")

        metrics.append({"epoch": ep, "train_loss": tr_loss, "test_loss": te_loss,
                        "acc": te_acc, "f1": te_f1, "qwk": te_qwk})

        score = te_qwk  # 以 QWK 作为早停指标（也可换成 acc/f1）
        if score > best:
            best, best_ep = score, ep
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            if args.save_model:
                torch.save(best_state, os.path.join(args.out_dir, "best_cell_tx_coral.pth"))
                print(f"[save] best @epoch {ep} -> {os.path.join(args.out_dir, 'best_cell_tx_coral.pth')}")

    print(f"[best] QWK={best:.2f}% @epoch {best_ep}")

    if args.save_metrics:
        pd.DataFrame(metrics).to_csv(os.path.join(args.out_dir, "metrics_by_epoch.csv"), index=False)
        print(f"[save] metrics -> {os.path.join(args.out_dir, 'metrics_by_epoch.csv')}")

if __name__ == "__main__":
    main()
