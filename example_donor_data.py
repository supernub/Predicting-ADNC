import scanpy as sc
import pandas as pd

path = "./training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad"
adata = sc.read_h5ad(path, backed="r")

# donor 元信息（每个 donor 一行）
cols = ["Donor ID", "ADNC", "Braak", "Thal", "CERAD", "Sex", "Age at Death"]
cols = [c for c in cols if c in adata.obs.columns]
df = adata.obs[cols].drop_duplicates(subset="Donor ID").head(5)

df.to_csv("example_donors.csv", index=False)
adata.file.close()
print("Wrote example_donors.csv")
