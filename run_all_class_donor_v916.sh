python mlp_donor_adnc_ver916.py \
  --data-path ../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad \
  --split-json split_donors.json \
  --donor-col "Donor ID" --adnc-col "ADNC" \
  #--use-x-scvi \
  --pool mean \
  --batch-donors 16 \
  --k-per-donor-train 2048\
  --k-per-donor-eval 8192 \
  --epochs 20 --patience 10 \
  --out-dir mil_a9_scvi\
  --save-model\ 
  --save-metrics
