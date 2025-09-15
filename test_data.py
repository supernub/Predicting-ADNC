import scanpy as sc

ad = sc.read_h5ad("../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad", backed="r")
print(ad.obs.columns[:20])   # 看前20个元数据列，确认 Class/Subclass/Supertype

# 统计 Supertype 的种类数
n_super = ad.obs["Supertype"].nunique()
print("Supertype 种类数:", n_super)

# 查看所有类别名
print("所有类别名字")
print(ad.obs["Supertype"].cat.categories if hasattr(ad.obs["Supertype"], "cat") else ad.obs["Supertype"].unique())

print(ad.obs["Supertype"].value_counts().head(20))   # 查看前20个最常见的 Supertype
