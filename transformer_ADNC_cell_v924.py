#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer (gene-as-token) + CORAL (ordinal regression) + MIL (donor aggregation)
DDP multi-GPU + AMP mixed precision, balanced donor sampler.

Usage (single GPU quick start):
  python transformer_ADNC_cell_v924.py \
    --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
    --split-json split_donors.json \
    --donor-col "Donor ID" --adnc-col "ADNC" \
    --top-k 512 --d-model 256 --layers 4 --heads 8 \
    --batch-donors 12 --k-per-donor-train 2048 --steps-per-epoch 80 \
    --epochs 30 --amp --out-dir runs_tx_coral_a9 --save-model --save-metrics

Usage (DDP with 4 GPUs):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 transformer_ADNC_cell_v924.py \
    --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
    --split-json split_donors.json \
    --donor-col "Donor ID" --adnc-col "ADNC" \
    --top-k 512 --d-model 256 --layers 4 --heads 8 \
    --batch-donors 16 --k-per-donor-train 2048 --steps-per-epoch 100 \
    --epochs 30 --ddp --amp --out-dir runs_tx_coral_a9_ddp --save-model --save-metrics
"""

import os, json, argparse
from datetime import timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch.cuda.amp import autocast, GradScaler

# -----------------------
# Config & args
# -----------------------
ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())

def get_args():
    ap = argparse.ArgumentParser("Transformer + CORAL + MIL for ADNC (DDP+AMP)")

    # data
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--split-json", required=True, help="donor split json with train_donors/test_donors")
    ap.add_argument("--donor-col", default="Donor ID")
    ap.add_argument("--adnc-col", default="ADNC")
    ap.add_argument("--filter-col", default=None)
    ap.add_argument("--filter-value", default=None)
    ap.add_argument("--icontains", action="store_true")

    # tokenization
    ap.add_argument("--top-k", type=int, default=512, help="Top-K nonzero genes per cell for tokens")
    ap.add_argument("--log1p", action="store_true", help="apply log1p to expression (default off; set if using raw)")
    # transformer
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    # training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-donors", type=int, default=12)
    ap.add_argument("--k-per-donor-train", type=int, default=2048)
    ap.add_argument("--k-per-donor-eval", type=int, default=6000)
    ap.add_argument("--steps-per-epoch", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)

    # DDP / AMP
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--amp", action="store_true")

    # output
    ap.add_argument("--out-dir", default="tx_coral_mil_runs")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--save-metrics", action="store_true")
    return ap.parse_args()

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# -----------------------
# Tokenizer (Top-K nonzeros per cell)
# -----------------------
def row_topk_tokens(row, k=512, log1p=False):
    """
    row: 1 x n_vars (scipy sparse or np array)
    return: gene_ids (L,), expr_vals (L,) with L<=k sorted by expr desc
    """
    if sp.issparse(row):
        row = row.tocsr()
        data = row.data
        cols = row.indices
    else:
        arr = np.asarray(row).ravel()
        nz = np.where(arr > 0)[0]
        cols = nz
        data = arr[nz]

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

def pad_batch_token_seqs(token_list, pad_token_id=-1, pad_expr=0.0):
    """
    token_list: list of (gene_ids np.int64[L], expr_vals np.float32[L])
    returns tensors: gene_ids (B,Lmax), expr_vals (B,Lmax), attn_mask (B,Lmax: True=mask pad)
    """
    B = len(token_list)
    L = max((len(g) for g, _ in token_list), default=1)
    gene_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    expr_vals = torch.full((B, L), pad_expr, dtype=torch.float32)
    attn_mask = torch.ones((B, L), dtype=torch.bool)
    for i, (g, e) in enumerate(token_list):
        if len(g) == 0: continue
        gene_ids[i, :len(g)] = torch.from_numpy(g)
        expr_vals[i, :len(e)] = torch.from_numpy(e)
        attn_mask[i, :len(g)] = False
    return gene_ids, expr_vals, attn_mask

# -----------------------
# MIL data containers (using tokenization on-the-fly)
# -----------------------
class DonorBags:
    """ hold cell indices per donor + donor labels (majority); provide subsampling """
    def __init__(self, X_mat, y_cell, donors_cell, donor_list, k_per_donor=None, rng=None):
        self.X = X_mat
        self.y_cell = np.asarray(y_cell, dtype=np.int64)
        self.donors_cell = np.asarray(donors_cell)
        self.donor_ids = sorted(list(set(donor_list)))
        self.k = k_per_donor
        self.rng = np.random.default_rng() if rng is None else rng

        self.idxs_by_donor = defaultdict(list)
        for i, d in enumerate(self.donors_cell):
            self.idxs_by_donor[d].append(i)

        maj = {}
        for d in self.donor_ids:
            idxs = self.idxs_by_donor[d]
            ys = [self.y_cell[i] for i in idxs]
            vals, cnts = np.unique(ys, return_counts=True)
            maj[d] = int(vals[np.argmax(cnts)])
        self.y_donor = maj

    def subsample_indices(self, idxs, k):
        if (k is None) or (len(idxs) <= k): return idxs
        return self.rng.choice(idxs, size=k, replace=False).tolist()

    def get_bag_indices(self, donor_id, k=None):
        idxs = self.idxs_by_donor[donor_id]
        use = self.subsample_indices(idxs, k)
        return use, self.y_donor[donor_id]

def collate_bags_tokens(donor_list, bags: DonorBags, X, k_cells, top_k=512, log1p=False):
    """
    Build one batch for donors:
      - tokenize each selected cell into (gene_ids, expr_vals)
      - pad to fixed length within this batch
      - bag_ptr marks donor boundaries
    returns:
      gene_ids: (Ncells, Lmax)
      expr_vals: (Ncells, Lmax)
      attn_mask: (Ncells, Lmax) True=mask pad
      bag_ptr: (Bdonor+1,)
      y_donors: (Bdonor,)
    """
    all_tokens = []
    bag_ptr, y_donors = [0], []
    n_cells_total = 0
    for d in donor_list:
        idxs, y_d = bags.get_bag_indices(d, k_cells)
        # tokenize cells of this donor
        for ridx in idxs:
            row = X[ridx]  # 1 x n_vars
            g, e = row_topk_tokens(row, k=top_k, log1p=log1p)
            all_tokens.append((g, e))
        n_cells_total += len(idxs)
        bag_ptr.append(n_cells_total)
        y_donors.append(y_d)
    gene_ids, expr_vals, attn_mask = pad_batch_token_seqs(all_tokens, pad_token_id=-1, pad_expr=0.0)
    bag_ptr = torch.tensor(bag_ptr, dtype=torch.long)
    y_donors = torch.tensor(y_donors, dtype=torch.long)
    return gene_ids, expr_vals, attn_mask, bag_ptr, y_donors

# -----------------------
# Transformer + FiLM + CORAL
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
    """ gene embedding + expression modulation (FiLM) """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.gene_emb = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.film = FiLM(d_model)
    def forward(self, gene_ids, expr_vals):
        # gene_ids:(B,L), expr_vals:(B,L)
        # pad token id = -1; 把 pad 的 gene_id 先夹到 [0, vocab_size-1]，mask 会处理
        gid = gene_ids.clamp(min=0)
        h = self.gene_emb(gid)                # (B,L,D)
        x = expr_vals.unsqueeze(-1)           # (B,L,1)
        h = self.film(h, x)
        return self.ln(h)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, heads=8, layers=4, ffn=512, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
    def forward(self, X, attn_mask):
        # X:(B,L,D), attn_mask:(B,L) True=mask pad
        B = X.size(0)
        cls = self.cls.expand(B, -1, -1)     # (B,1,D)
        X = torch.cat([cls, X], dim=1)       # (B,1+L,D)
        # 扩展 mask：前置 CLS 位置不 mask
        pad = torch.zeros((B,1), dtype=attn_mask.dtype, device=attn_mask.device)
        m = torch.cat([pad, attn_mask], dim=1)  # (B,1+L)
        H = self.enc(X, src_key_padding_mask=m) # (B,1+L,D)
        return H[:,0,:]                        # CLS

class CoralHead(nn.Module):
    def __init__(self, in_dim, K=3, hid=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, K)
        )
    def forward(self, z):  # z:(B,D)
        return self.net(z) # (B,K)

def coral_loss(logits, y, K=3):
    # logits:(B,K) for P(y>k); y:(B,)
    y = y.unsqueeze(1)
    ks = torch.arange(K, device=logits.device).unsqueeze(0)
    targets = (y > ks).float()
    return F.binary_cross_entropy_with_logits(logits, targets)

@torch.no_grad()
def coral_decode(logits):  # (B,K)
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1)  # class id in [0..K]

class CellTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=8, layers=4, ffn=512, dropout=0.1, K=3):
        super().__init__()
        self.embed = GeneTokenEmbed(vocab_size, d_model)
        self.backbone = TransformerEncoder(d_model, heads, layers, ffn, dropout)
        self.head = CoralHead(d_model, K=K, hid=128, p=dropout)

    def forward(self, gene_ids, expr_vals, attn_mask):
        # gene_ids/expr_vals:(Ncells, L), attn_mask:(Ncells, L)
        X = self.embed(gene_ids, expr_vals)          # (N,L,D)
        z = self.backbone(X, attn_mask)              # (N,D)  cell-level
        logits_k = self.head(z)                      # (N,K)
        return logits_k

# donor aggregation for CORAL logits
def aggregate_donor_coral(logits_k, bag_ptr):
    """
    logits_k: (Ncells, K)
    bag_ptr: (B+1,) cumulative counts
    return (B,K) donor-level logits (mean of cell logits)
    """
    B = bag_ptr.numel() - 1
    outs = []
    for b in range(B):
        s, e = bag_ptr[b].item(), bag_ptr[b+1].item()
        outs.append(logits_k[s:e].mean(dim=0, keepdim=True))
    return torch.vstack(outs) if outs else torch.empty(0, logits_k.size(1), device=logits_k.device)

# -----------------------
# Balanced donor sampler (with replacement, fixed steps/epoch)
# -----------------------
class BalancedDonorBatchSampler(IterableDataset):
    def __init__(self, donors_by_class, batch_donors=12, seed=42, classes=(0,1,2,3), steps_per_epoch=80):
        super().__init__()
        self.donors_by_class = {int(c): list(v) for c, v in donors_by_class.items()}
        self.classes = [c for c in classes if len(self.donors_by_class.get(c, [])) > 0]
        if len(self.classes) == 0:
            raise ValueError("No classes available.")
        self.per_class = max(1, batch_donors // len(self.classes))
        if batch_donors % len(self.classes) != 0:
            print(f"[warn] batch_donors={batch_donors} not divisible by #classes={len(self.classes)}; "
                  f"use per_class={self.per_class}, effective={self.per_class*len(self.classes)}")
        self.batch_donors = self.per_class * len(self.classes)
        self.steps_per_epoch = int(steps_per_epoch)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            for c in self.classes:
                pool = self.donors_by_class[c]
                pick = self.rng.choice(pool, size=self.per_class, replace=True).tolist()
                batch.extend(pick)
            self.rng.shuffle(batch)
            yield batch

def shard_for_rank(seq, rank, world):
    return seq[rank::world]

# -----------------------
# Eval
# -----------------------
@torch.no_grad()
def evaluate(model, donor_chunks, bags, X, device, k_eval, top_k, log1p, amp=False, main=True):
    if not main:
        return 0.0, 0.0
    model.eval()
    from sklearn.metrics import accuracy_score, f1_score
    y_true, y_pred = [], []
    for donor_batch in donor_chunks:
        gene_ids, expr_vals, attn_mask, bag_ptr, y_d = collate_bags_tokens(
            donor_batch, bags, X, k_eval, top_k=top_k, log1p=log1p
        )
        if gene_ids.numel() == 0: continue
        gene_ids = gene_ids.to(device, non_blocking=True)
        expr_vals = expr_vals.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        bag_ptr = bag_ptr.to(device)

        if amp:
            with autocast(dtype=torch.float16):
                cell_logits = model(gene_ids, expr_vals, attn_mask)
        else:
            cell_logits = model(gene_ids, expr_vals, attn_mask)

        donor_logits = aggregate_donor_coral(cell_logits, bag_ptr)  # (B,K)
        pred = coral_decode(donor_logits).cpu()                     # (B,)
        y_true.extend(y_d.tolist())
        y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred) * 100.0
    f1  = f1_score(y_true, y_pred, average="weighted") * 100.0
    return acc, f1

def donor_label_distribution(bags, name, main=True):
    if not main: return
    d2y = bags.y_donor
    donors_by_cls = defaultdict(list)
    for d, yy in d2y.items(): donors_by_cls[yy].append(d)
    cells_by_cls = Counter()
    for d, idxs in bags.idxs_by_donor.items():
        if d in d2y: cells_by_cls[d2y[d]] += len(idxs)
    print(f"[{name}] donors per class:", {k: len(v) for k, v in sorted(donors_by_cls.items())})
    print(f"[{name}] cells  per class:", dict(sorted(cells_by_cls.items())))

# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # DDP init
    local_rank, world_size, is_main = 0, 1, True
    if args.ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=3600))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        is_main = (local_rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # donor split
    with open(args.split_json, "r") as f:
        sp = json.load(f)
    train_donors = set(sp.get("train_donors") or sp.get("train") or [])
    test_donors  = set(sp.get("test_donors")  or sp.get("test")  or [])
    if not train_donors or not test_donors:
        raise ValueError("split json must contain train_donors/test_donors")

    # read h5ad
    ad = sc.read_h5ad(args.data_path, backed="r")  # backed=r 节省内存
    adnc_col = args.adnc_col
    if adnc_col not in ad.obs_keys():
        if "Overall AD neuropathological Change" in ad.obs_keys():
            adnc_col = "Overall AD neuropathological Change"
        else:
            raise KeyError("ADNC column not found")

    # optional filter
    if args.filter_col and args.filter_value and args.filter_col in ad.obs_keys():
        s = ad.obs[args.filter_col].astype(str)
        mask = s.str.contains(args.filter_value, case=False, regex=False) if args.icontains else (s == args.filter_value)
        idx_keep = np.where(mask.values)[0]
    else:
        idx_keep = np.arange(ad.n_obs)

    # keep valid ADNC
    adnc = ad.obs[adnc_col].astype(str).values
    keep_valid = np.where(np.isin(adnc, list(ADNC_ALLOWED)))[0]
    idx_all = np.intersect1d(idx_keep, keep_valid, assume_unique=False)

    donors = ad.obs[args.donor_col].astype(str).values[idx_all]
    y_cell = np.array([ADNC_MAP[s] for s in ad.obs[adnc_col].astype(str).values[idx_all]], dtype=np.int64)

    donors_in_file = set(np.unique(donors))
    train_d = sorted(list(donors_in_file & train_donors))
    test_d  = sorted(list(donors_in_file & test_donors))
    if is_main:
        print(f"[donors in file] train={len(train_d)} test={len(test_d)}")

    # build matrix view limited to idx_all
    # backed 模式下 ad.X 支持切片（按行），这里保留原 ad 以便 row 索引
    class XView:
        def __init__(self, ad, idx_all):
            self.ad = ad; self.idx = idx_all
            self.n_vars = ad.n_vars
        def __getitem__(self, ridx):
            # ridx 是全局 row id（相对于原 ad），这里确保传进来的是原全局索引
            return self.ad.X[ridx]
    X = XView(ad, idx_all)

    # DonorBags（注意：这里传入 donors/y_cell 是 idx_all 对应的切片）
    bags_tr = DonorBags(X, y_cell, donors, donor_list=train_d,
                        k_per_donor=args.k_per_donor_train, rng=np.random.default_rng(args.seed))
    bags_te = DonorBags(X, y_cell, donors, donor_list=test_d,
                        k_per_donor=args.k_per_donor_eval, rng=np.random.default_rng(args.seed+1))

    donor_label_distribution(bags_tr, "train", is_main)
    donor_label_distribution(bags_te, "test",  is_main)

    # donors_by_class for balanced sampler
    donor2y = bags_tr.y_donor
    donors_by_class = defaultdict(list)
    for d, yy in donor2y.items():
        donors_by_class[int(yy)].append(d)

    # model
    vocab_size = ad.n_vars  # gene_id == 列索引
    model = CellTransformer(vocab_size,
                            d_model=args.d_model, heads=args.heads,
                            layers=args.layers, ffn=args.ffn,
                            dropout=args.dropout, K=3).to(device)

    # loss
    ce = None  # 我们使用 coral_loss 代替 CE
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if args.amp else None

    # DDP wrap
    if args.ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # sampler
    sampler = BalancedDonorBatchSampler(
        donors_by_class, batch_donors=args.batch_donors,
        seed=args.seed, steps_per_epoch=args.steps_per_epoch
    )

    def train_one_epoch():
        model.train()
        run_loss, steps = 0.0, 0
        for donor_batch in sampler:
            if args.ddp and world_size > 1:
                donor_batch = shard_for_rank(donor_batch, local_rank, world_size)
                if len(donor_batch) == 0: continue

            gene_ids, expr_vals, attn_mask, bag_ptr, y_d = collate_bags_tokens(
                donor_batch, bags_tr, X, args.k_per_donor_train, top_k=args.top_k, log1p=args.log1p
            )
            if gene_ids.numel() == 0: continue

            gene_ids = gene_ids.to(device, non_blocking=True)
            expr_vals = expr_vals.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            bag_ptr = bag_ptr.to(device)
            y_d = y_d.to(device)

            opt.zero_grad(set_to_none=True)
            if args.amp:
                with autocast(dtype=torch.float16):
                    cell_logits = model(gene_ids, expr_vals, attn_mask)    # (Ncells,K)
                    donor_logits = aggregate_donor_coral(cell_logits, bag_ptr)  # (B,K)
                    loss = coral_loss(donor_logits, y_d, K=3)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                cell_logits = model(gene_ids, expr_vals, attn_mask)
                donor_logits = aggregate_donor_coral(cell_logits, bag_ptr)
                loss = coral_loss(donor_logits, y_d, K=3)
                loss.backward()
                opt.step()

            run_loss += float(loss.detach().cpu())
            steps += 1
        return run_loss / max(1, steps)

    # eval donor chunks
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]
    te_chunks = list(chunks(test_d, args.batch_donors))

    best, best_ep, patience, pat = -np.inf, -1, 10, 10
    metrics = []

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch()
        da, df = evaluate(model, te_chunks, bags_te, X, device,
                          args.k_per_donor_eval, args.top_k, args.log1p,
                          amp=args.amp, main=is_main)
        if is_main:
            print(f"[epoch {ep}] train_loss={tr_loss:.4f} | DONOR acc={da:.2f}% f1={df:.2f}%")
            metrics.append({"epoch": ep, "train_loss": tr_loss, "donor_acc": da, "donor_f1": df})
            score = da
            if score > best:
                best, best_ep, pat = score, ep, patience
                # 保存不带 module.
                state = (model.module.state_dict() if args.ddp else model.state_dict())
                best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            else:
                pat -= 1
                if pat <= 0:
                    print(f"[early stop] best acc={best:.2f}% @epoch {best_ep}")
                    break

    if is_main and best > -1e9:
        # 恢复最佳并再次评估
        if args.ddp: model.module.load_state_dict(best_state)
        else:        model.load_state_dict(best_state)
        da, df = evaluate(model, te_chunks, bags_te, X, device,
                          args.k_per_donor_eval, args.top_k, args.log1p,
                          amp=args.amp, main=is_main)
        print(f"[best @epoch {best_ep}] DONOR acc={da:.2f}% f1={df:.2f}%")

        if args.save_metrics:
            pd.DataFrame(metrics).to_csv(os.path.join(args.out_dir, "metrics_by_epoch.csv"), index=False)
            print(f"[save] metrics -> {os.path.join(args.out_dir, 'metrics_by_epoch.csv')}")
        if args.save_model:
            torch.save(best_state, os.path.join(args.out_dir, "best_tx_coral_mil.pth"))
            print(f"[save] model -> {os.path.join(args.out_dir, 'best_tx_coral_mil.pth')}")

    # cleanup
    if args.ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
