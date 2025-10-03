#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

def choose_indices_stratified(obs_df, strata_col, n_total, seed=42):
    rng = np.random.default_rng(seed)
    groups = obs_df.groupby(strata_col).indices  # dict: level -> row indices
    # 每个组目标样本数近似按组大小比例分配，但至少各组取 1
    sizes = {k: len(v) for k, v in groups.items()}
    total = sum(sizes.values())
    # 先按比例分配
    alloc = {k: max(1, int(round(n_total * sizes[k] / total))) for k in sizes}
    # 修正分配数和 n_total 一致
    diff = n_total - sum(alloc.values())
    keys = list(alloc.keys())
    i = 0
    while diff != 0 and len(keys) > 0:
        k = keys[i % len(keys)]
        if diff > 0:
            alloc[k] += 1; diff -= 1
        else:
            if alloc[k] > 1:
                alloc[k] -= 1; diff += 1
        i += 1
    # 抽样
    sel = []
    for k, idxs in groups.items():
        take = min(alloc[k], len(idxs))
        choice = rng.choice(idxs, size=take, replace=False)
        sel.append(choice)
    if len(sel) == 0:
        return np.array([], dtype=int)
    return np.concatenate(sel)

def choose_indices_random(n_cells, n_total, seed=42):
    rng = np.random.default_rng(seed)
    take = min(n_total, n_cells)
    return rng.choice(np.arange(n_cells), size=take, replace=False)

def main():
    ap = argparse.ArgumentParser("Show adata cell examples (no scVI, from adata.X)")
    ap.add_argument("--data-path", required=True, help=".h5ad path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--strata", type=str, default=None, help="obs column to stratify by (e.g., Class/Subclass/Supertype/'Donor ID')")
    ap.add_argument("--genes", type=str, default="", help="comma-separated gene symbols present in var_names")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[info] reading h5ad (backed='r'): {args.data_path}")
    ad = sc.read_h5ad(args.data_path, backed="r")

    # 解析基因列表
    genes = [g.strip() for g in args.genes.split(",") if g.strip()] if args.genes else []
    # 将 var_names 转成 numpy，便于查找位置
    var_names = np.array(ad.var_names)

    # 选细胞索引
    obs_df = ad.obs.copy()  # backed='r' 下 obs 是内存 DataFrame
    if args.strata and args.strata in obs_df.columns:
        print(f"[info] stratified sampling by: {args.strata}")
        sel_idx = choose_indices_stratified(obs_df, args.strata, args.n_cells, seed=args.seed)
    else:
        print(f"[info] random sampling without strata")
        sel_idx = choose_indices_random(ad.n_obs, args.n_cells, seed=args.seed)

    sel_idx = np.sort(sel_idx)
    print(f"[info] sampled cells: {len(sel_idx)}")

    # 导出 obs
    obs_sample = obs_df.iloc[sel_idx].copy()
    obs_csv = os.path.join(args.out_dir, "obs_preview.csv")
    obs_sample.to_csv(obs_csv)
    print(f"[save] obs -> {obs_csv}")

    # 分层计数（可选）
    if args.strata and args.strata in obs_df.columns:
        counts = obs_sample[args.strata].value_counts().rename_axis(args.strata).reset_index(name="n_cells")
        counts_csv = os.path.join(args.out_dir, "obs_counts_by_strata.csv")
        counts.to_csv(counts_csv, index=False)
        print(f"[save] counts(by {args.strata}) -> {counts_csv}")

    # 导出选定基因表达（少量示例，便于分享）
    if genes:
        # 过滤掉不在 var_names 的基因
        exist_mask = np.isin(genes, var_names)
        missing = [g for g, ok in zip(genes, exist_mask) if not ok]
        if missing:
            print(f"[warn] genes not found in var_names: {missing}")
        genes_exist = [g for g in genes if g in set(var_names)]
        if genes_exist:
            # 找列索引
            col_idx = np.where(np.isin(var_names, genes_exist))[0]
            # 从 adata.X 取子矩阵（注意 backed='r' 支持切片）
            X_sub = ad.X[sel_idx, :]    # 先取行（仍是稀疏）
            X_sub = X_sub[:, col_idx]   # 再取列
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            expr_df = pd.DataFrame(X_sub, columns=var_names[col_idx], index=obs_sample.index)
            expr_csv = os.path.join(args.out_dir, "expr_selected_genes.csv")
            expr_df.to_csv(expr_csv)
            print(f"[save] expr(selected genes) -> {expr_csv}")
        else:
            print("[info] no valid genes to export (all missing).")

    # 画 UMAP（如果有）
    if "X_umap" in ad.obsm_keys():
        print("[info] plotting UMAP for sampled cells...")
        umap = ad.obsm["X_umap"][sel_idx, :]
        # 颜色：用 ADNC 或者 strata 如果有；否则不着色
        color_col = None
        for candidate in ["ADNC", args.strata, "Supertype", "Subclass", "Class"]:
            if candidate and candidate in obs_sample.columns:
                color_col = candidate; break

        plt.figure(figsize=(6, 6))
        if color_col:
            # 离散上色（matplotlib 默认 colormap）
            vals = obs_sample[color_col].astype(str).values
            cats, inv = np.unique(vals, return_inverse=True)
            sca = plt.scatter(umap[:, 0], umap[:, 1], s=3, c=inv, alpha=0.8)
            plt.title(f"UMAP (sampled) colored by {color_col}")
            # 简单做个图例（最多显示前 15 类，避免过长）
            for i, c in enumerate(cats[:15]):
                plt.scatter([], [], c=sca.cmap(sca.norm(i)), label=c, s=10)
            if len(cats) > 15:
                plt.legend(title=f"{color_col} (top 15 shown)", markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                plt.legend(title=color_col, markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            plt.scatter(umap[:, 0], umap[:, 1], s=3, alpha=0.8)
            plt.title("UMAP (sampled)")

        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "umap_sample.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[save] UMAP -> {fig_path}")
    else:
        print("[info] adata.obsm['X_umap'] not found; skip UMAP plot.")

    print("[done] All artifacts written to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
