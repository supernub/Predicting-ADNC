#!/usr/bin/env python3
"""
python make_donor_split.py \
  --h5ads ./training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
          ./training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad \
  --out split_donors.json \
  --test-size 0.2 \
  --seed 42

"""

import argparse, json, numpy as np, scanpy as sc

def get_donors(h5):
    ad = sc.read_h5ad(h5, backed="r")
    donors = ad.obs["Donor ID"].astype(str).unique().tolist()
    ad.file.close()
    return donors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ads", nargs="+", required=True, help="list of .h5ad paths (A9, MTG, ...)")
    ap.add_argument("--out", default="split_donors.json")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 收集所有 donor
    all_donors = set()
    for p in args.h5ads:
        all_donors.update(get_donors(p))
    all_donors = sorted(all_donors)
    n = len(all_donors)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_donors)

    n_test = max(1, int(round(n * args.test_size)))
    test_ids = sorted(all_donors[:n_test])
    train_ids = sorted(all_donors[n_test:])

    split = {"seed": args.seed, "test_size": args.test_size,
             "train_donors": train_ids, "test_donors": test_ids}

    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[done] donors: {n} -> train {len(train_ids)}, test {len(test_ids)}")
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
