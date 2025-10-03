#!/usr/bin/env python3
#按照DONOR 进行分类83/21 例子
import argparse, json, numpy as np, scanpy as sc

def get_donors(h5ad_path, donor_col="Donor ID"):
    ad = sc.read_h5ad(h5ad_path, backed="r")
    donors = ad.obs[donor_col].astype(str).unique().tolist()
    ad.file.close()
    return donors

def main():
    ap = argparse.ArgumentParser("Make donor split (80/20)")
    ap.add_argument("--h5ads", nargs="+", required=True, help="one or multiple .h5ad paths")
    ap.add_argument("--donor-col", default="Donor ID")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="split_donors.json")
    args = ap.parse_args()

    all_donors = set()
    for p in args.h5ads:
        all_donors.update(get_donors(p, args.donor_col))
    all_donors = sorted(all_donors)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_donors)
    n_test = max(1, int(round(len(all_donors) * args.test_size)))
    test_donors = sorted(all_donors[:n_test])
    train_donors = sorted(all_donors[n_test:])

    split = {
        "seed": args.seed,
        "test_size": args.test_size,
        "donor_col": args.donor_col,
        "train_donors": train_donors,
        "test_donors": test_donors,
    }
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[done] donors={len(all_donors)} -> train={len(train_donors)} test={len(test_donors)}")
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
