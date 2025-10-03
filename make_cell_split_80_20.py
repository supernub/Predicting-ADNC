#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python make_cell_split_80_20.py \
  --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
  --stratify \
  --out-json cell_split_A9_80_20.json

"""

import argparse, json, os
import numpy as np
import scanpy as sc

ADNC_MAP = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
ADNC_ALLOWED = set(ADNC_MAP.keys())

def main():
    ap = argparse.ArgumentParser("make cell-level 80/20 split (optionally stratified by ADNC)")
    ap.add_argument("--data-path", required=True, help=".h5ad path")
    ap.add_argument("--adnc-col", default="ADNC")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--stratify", action="store_true", help="stratify by ADNC (recommended)")
    ap.add_argument("--out-json", default="cell_split_80_20.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ad = sc.read_h5ad(args.data_path, backed="r")
    # 找到 ADNC 列
    adnc_col = args.adnc_col
    if adnc_col not in ad.obs_keys():
        if "Overall AD neuropathological Change" in ad.obs_keys():
            adnc_col = "Overall AD neuropathological Change"
        else:
            raise KeyError(f"ADNC column '{args.adnc_col}' not found in obs.")

    # 取有效 ADNC
    adnc = ad.obs[adnc_col].astype(str)
    valid = adnc.isin(ADNC_ALLOWED).values
    idx_all = np.where(valid)[0]
    labels = adnc.values[valid]
    y = np.array([ADNC_MAP[s] for s in labels], dtype=np.int64)

    n = len(idx_all)
    n_test = int(round(n * args.test_size))

    if args.stratify:
        # 按 y 分层抽样 test
        test_idx = []
        for cls in np.unique(y):
            cls_pos = np.where(y == cls)[0]
            k = int(round(len(cls_pos) * args.test_size))
            pick = rng.choice(cls_pos, size=k, replace=False).tolist()
            test_idx.extend(pick)
        test_idx = np.array(test_idx, dtype=int)
        # 兜底：若总量不等于 n_test，随机补/裁
        if len(test_idx) < n_test:
            remain = np.setdiff1d(np.arange(n), test_idx, assume_unique=False)
            extra = rng.choice(remain, size=(n_test - len(test_idx)), replace=False)
            test_idx = np.concatenate([test_idx, extra])
        elif len(test_idx) > n_test:
            test_idx = rng.choice(test_idx, size=n_test, replace=False)
    else:
        perm = rng.permutation(n)
        test_idx = perm[:n_test]

    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    tr_rel = np.where(~mask)[0]
    te_rel = np.where(mask)[0]

    # 转回到全体细胞的绝对索引
    train_indices = idx_all[tr_rel].tolist()
    test_indices  = idx_all[te_rel].tolist()

    out = {
        "data_path": os.path.abspath(args.data_path),
        "adnc_col": adnc_col,
        "seed": args.seed,
        "test_size": args.test_size,
        "stratify": bool(args.stratify),
        "train_indices": train_indices,
        "test_indices": test_indices
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"[ok] split saved -> {args.out_json}")
    print(f"train cells: {len(train_indices)} | test cells: {len(test_indices)}")

if __name__ == "__main__":
    main()
