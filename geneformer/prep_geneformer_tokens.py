# prep_geneformer_tokens.py
import os, json, argparse
import numpy as np
import scanpy as sc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def load_split(split_json):
    with open(split_json) as f:
        d = json.load(f)
    # 兼容不同命名
    train_idx = d.get("train_indices") or d.get("train")
    val_idx   = d.get("val_indices") or d.get("valid_indices") or d.get("test_indices") or d.get("val") or d.get("test")
    if train_idx is None or val_idx is None:
        raise ValueError("split_json 里找不到 train/val 索引，请检查键名（train_indices / val_indices / test_indices 等）")
    return list(map(int, train_idx)), list(map(int, val_idx))

def make_label_encoder(labels, label_order=None):
    if label_order:
        # 用给定顺序（序数映射）
        classes = [s.strip() for s in label_order.split(",")]
        le = LabelEncoder()
        le.classes_ = np.array(classes, dtype=object)
        y = le.transform(labels)
    else:
        le = LabelEncoder()
        y = le.fit_transform(labels)
    return le, y

def topk_tokens_for_cell(x_dense_row, gene_symbols, gene2id, top_k):
    # x_dense_row: (n_genes,) float
    # 取前 top_k 个表达最高的基因（非零）
    nz = np.nonzero(x_dense_row)[0]
    if nz.size == 0:
        return []
    vals = x_dense_row[nz]
    order = nz[np.argsort(-vals)]
    order = order[:top_k]
    tokens = []
    for gi in order:
        gs = gene_symbols[gi]
        tid = gene2id.get(gs)
        if tid is not None:
            tokens.append(int(tid))
    return tokens

def dump_list_of_dicts(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import pickle as pkl
    with open(path, "wb") as f:
        pkl.dump(arr, f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--label_col", default="ADNC")
    ap.add_argument("--label_order", default="Not AD,Low,Intermediate,High")
    ap.add_argument("--gene_vocab_json", required=True, help="Geneformer 的基因→token词表 json")
    ap.add_argument("--top_k", type=int, default=2048, help="每个cell取前K个表达基因作为token")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    print(f"Reading {args.h5ad} (backed='r') ...")
    ad = sc.read_h5ad(args.h5ad, backed='r')
    n_cells, n_genes = ad.shape
    print("cells:", n_cells, "genes:", n_genes)

    # 基因名：优先 gene_symbol，其次 var_names
    if "gene_symbol" in ad.var.columns:
        gene_symbols = ad.var["gene_symbol"].astype(str).to_numpy()
    else:
        gene_symbols = ad.var_names.to_numpy().astype(str)

    # 载入 gene→token 词表
    with open(args.gene_vocab_json) as f:
        gene2id = json.load(f)

    # split
    train_idx, val_idx = load_split(args.split_json)

    # 标签
    labels_raw = ad.obs[args.label_col].astype(str).to_numpy()
    le, y_all  = make_label_encoder(labels_raw, args.label_order)
    mapping = {cls: int(i) for i, cls in enumerate(le.classes_)}

    # 导出类别映射
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "label_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    # 提示：backed='r' 下 ad.X[i,:].toarray() 需要小心效率；我们只遍历 split 索引
    def build_split(split_idx, name):
        data = []
        for i in tqdm(split_idx, desc=f"tokenizing {name}", mininterval=0.5):
            x = ad.X[i].toarray().ravel() if hasattr(ad.X, "toarray") else np.asarray(ad.X[i]).ravel()
            toks = topk_tokens_for_cell(x, gene_symbols, gene2id, args.top_k)
            if len(toks) == 0:
                continue
            item = {"input_ids": toks, "labels": int(y_all[i])}
            data.append(item)
        outp = os.path.join(args.out_dir, f"{name}.tokenized.pkl")
        dump_list_of_dicts(outp, data)
        print(name, "saved:", outp, "items:", len(data))

    build_split(train_idx, "train")
    build_split(val_idx,   "val")

    # 保存一些元信息
    meta = {
        "h5ad": os.path.abspath(args.h5ad),
        "top_k": args.top_k,
        "label_col": args.label_col,
        "label_order": args.label_order,
        "mapping": mapping
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
