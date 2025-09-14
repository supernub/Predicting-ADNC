import scanpy as sc
import pandas as pd
import numpy as np

path = "./training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad"
adata = sc.read_h5ad(path, backed="r")

# 随机抽 100 个细胞
idx = np.random.choice(adata.n_obs, size=100, replace=False)

# 取元数据
obs_cols = ["Donor ID", "Supertype", "ADNC", "Sex", "Age at Death"]
obs_cols = [c for c in obs_cols if c in adata.obs.columns]
df_obs = adata.obs.iloc[idx][obs_cols].astype(str)

# 取 latent 表达
Z = adata.obsm["X_scVI"][idx, :10]  # 只取前 10 维，展示用
df_lat = pd.DataFrame(Z, columns=[f"scVI_{i}" for i in range(Z.shape[1])], index=df_obs.index)

# 合并
df = pd.concat([df_obs, df_lat], axis=1)
df.to_csv("example_cells.csv", index=False)

adata.file.close()
print("Wrote example_cells.csv")
