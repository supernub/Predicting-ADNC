# dataset_varlen.py  —— memmap 共享 + 惰性版（覆盖原文件）
import os, numpy as np
from typing import List, Optional
import torch
import scanpy as sc
from torch.utils.data import Dataset
from scipy import sparse
from typing import List, Dict, Optional, Union, Tuple


PAD_IDX = 0

class CSRMemmapDataset(Dataset):
    """
    基于 CSR 三数组 (indptr/indices/data) 的内存映射，多进程共享 OS pagecache。
    每次 __getitem__ 只切出该 cell 的 [start:end) 段返回。
    """
    def __init__(self, memmap_dir: str, labels: np.ndarray, cell_indices: List[int]):
        self.indptr  = np.load(os.path.join(memmap_dir,'indptr.npy'),  mmap_mode='r')
        self.indices = np.load(os.path.join(memmap_dir,'indices.npy'), mmap_mode='r')
        self.data    = np.load(os.path.join(memmap_dir,'data.npy'),    mmap_mode='r')
        with open(os.path.join(memmap_dir,'shape.txt'),'r') as f:
            n, m = map(int, f.read().strip().split())
        self.n_cells, self.n_genes = n, m
        self.labels_all = labels
        self.ids = np.asarray(cell_indices, dtype=np.int64)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        ridx = int(self.ids[i])
        start, end = int(self.indptr[ridx]), int(self.indptr[ridx+1])
        idx = self.indices[start:end].astype(np.int64) + 1   # 0 留给 PAD
        val = self.data[start:end].astype(np.float32)
        seq = torch.from_numpy(idx)
        vals = torch.from_numpy(val)
        label = torch.as_tensor(self.labels_all[ridx], dtype=torch.long)
        return seq, vals, label

class VariableLengthSequenceDataset(Dataset):
    """
    普通惰性模式（非 memmap）：引用 X 的 CSR，不在 __init__ 展开所有样本。
    """
    def __init__(self, X, labels: np.ndarray, cell_indices: List[int]):
        assert sparse.issparse(X), "X must be a scipy sparse matrix"
        self.X = X.tocsr(copy=False)
        self.labels_all = labels
        self.ids = np.asarray(cell_indices, dtype=np.int64)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        ridx = int(self.ids[i])
        row = self.X.getrow(ridx)
        seq = torch.from_numpy(row.indices.astype(np.int64) + 1)
        vals = torch.from_numpy(row.data.astype(np.float32))
        label = torch.as_tensor(self.labels_all[ridx], dtype=torch.long)
        return seq, vals, label

def pad_collate(batch, pad_idx: int = PAD_IDX):
    seqs, vals, labels = zip(*batch)
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    vals = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels, dim=0)
    attn_mask = (seqs == pad_idx)   # True 表示 PAD
    return seqs, vals, labels, attn_mask



def _infer_label_mapping(labels_col: np.ndarray, explicit_order: Optional[List[str]] = None) -> Dict[Union[str,int], int]:
    """
    returns mapping original_label -> ordinal {0..K-1}
    If explicit_order is provided, map by that order.
    Otherwise, try a sensible default for ADNC, else sort unique.
    """
    if explicit_order is not None:
        order = [s.strip() for s in explicit_order]
        uniq = list(dict.fromkeys(order))  # preserve order
    else:
        # heuristics for ADNC names
        uniq_str = np.unique(labels_col.astype(str))
        # common ADNC orders
        candidates = [
            ["Not AD", "Low", "Intermediate", "High"],
            ["A0","A1","A2","A3"],
            ["0","1","2","3"],
        ]
        matched = None
        for cand in candidates:
            if set(uniq_str) <= set(cand):
                matched = cand
                break
        if matched is not None:
            uniq = matched
        else:
            # fallback: sort unique by natural order
            try:
                uniq = sorted(uniq_str, key=lambda x: float(x))
            except:
                uniq = sorted(list(uniq_str))

    mapping = {k:i for i,k in enumerate(uniq)}
    return mapping


def load_labels_and_split_only(h5ad_path: str, split_json_path: str,
                               label_col: str, label_order=None):
    import json, scanpy as sc, numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    adata = sc.read_h5ad(h5ad_path, backed='r')  # 只读映射
    n_cells, n_genes = adata.shape

    with open(split_json_path) as f:
        split = json.load(f)

    train_idx = None
    val_idx = None

    # 索引命名（train + val/valid/test）
    if "train_indices" in split:
        train_idx = split["train_indices"]
        for k in ["val_indices", "valid_indices", "test_indices"]:
            if k in split:
                val_idx = split[k]
                break

    # 条形码命名
    if train_idx is None and "train_barcodes" in split:
        name_to_idx = {name: i for i, name in enumerate(adata.obs_names)}
        train_idx = [name_to_idx[b] for b in split["train_barcodes"] if b in name_to_idx]
        for k in ["val_barcodes", "valid_barcodes", "test_barcodes"]:
            if k in split:
                val_idx = [name_to_idx[b] for b in split[k] if b in name_to_idx]
                break

    # 计算 y（用于 Stratified）
    labels_raw = adata.obs[label_col].astype(str).to_numpy()
    if label_order:
        order = [s.strip() for s in label_order]
    else:
        order = sorted(set(labels_raw))
    mapping = {k: i for i, k in enumerate(order)}
    y = np.array([mapping.get(v, -1) for v in labels_raw], dtype=np.int64)
    if (y < 0).any():
        bad = sorted(set(labels_raw[y < 0].tolist()))
        raise ValueError(f"Found labels not in label_order: {bad}")

    # 若没有 val，就按 20% 分层切
    if val_idx is None:
        idx = np.arange(n_cells)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, va = next(sss.split(idx, y))
        train_idx, val_idx = tr.tolist(), va.tolist()
    else:
        train_idx = list(map(int, train_idx))
        val_idx = list(map(int, val_idx))

    num_classes = len(order)
    return y, train_idx, val_idx, n_genes, num_classes, mapping


#
def load_h5ad_with_split(
    h5ad_path: str,
    split_json_path: str,
    label_col: str,
    label_order: Optional[List[str]] = None
):
    """
    split json format (支持多种):
    {
      "train_indices": [0,2,5,...],
      "val_indices":   [1,3,...]
    }
    或者
    {
      "train_barcodes": ["AAAC...-1", ...],
      "val_barcodes":   ["AAAG...-1", ...]
    }
    """
    print(f"Reading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X  # expected csr
    obs = adata.obs

    with open(split_json_path, "r") as f:
        split = json.load(f)

    if "train_indices" in split and "val_indices" in split:
        train_idx = split["train_indices"]
        val_idx = split["val_indices"]
    elif "train_barcodes" in split and "val_barcodes" in split:
        name_to_idx = {name:i for i,name in enumerate(adata.obs_names)}
        train_idx = [name_to_idx[b] for b in split["train_barcodes"] if b in name_to_idx]
        val_idx   = [name_to_idx[b] for b in split["val_barcodes"]   if b in name_to_idx]
    else:
        raise ValueError("Split JSON must contain either indices or barcodes (train/val).")

    raw_labels = obs[label_col].to_numpy()
    if raw_labels.dtype.kind in ['i','u','f']:
        # numeric labels already; map to 0..K-1 by sorted unique
        uniq = sorted(list(np.unique(raw_labels)))
        mapping = {k:i for i,k in enumerate(uniq)}
    else:
        mapping = _infer_label_mapping(raw_labels, explicit_order=label_order)

    y = np.array([mapping[str(v)] if not isinstance(v,(int,np.integer,float,np.floating)) else mapping[v]
                  for v in raw_labels], dtype=np.int64)

    num_classes = len(set(mapping.values()))
    num_genes = X.shape[1]

    return X, y, train_idx, val_idx, num_genes, num_classes, mapping
