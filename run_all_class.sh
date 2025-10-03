#!/bin/bash
DATA=../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad
SPLIT=split_donors.json
OUTDIR=class_results

mkdir -p $OUTDIR

for CLASS in "Neuronal: Glutamatergic" "Neuronal: GABAergic" "Non-neuronal and Non-neural"
do
  echo "=== Running $CLASS ==="
  python mlp_adnc_per_celltype_split.py \
    --data-path "$DATA" \
    --split-json "$SPLIT" \
    --cell-type "$CLASS" \
    --celltype-col "Class" \
    --adnc-col "ADNC" \
    --donor-col "Donor ID" \
#    --use-x-scvi \
    --epochs 10 --batch-size 256 \
    --out-dir "$OUTDIR/$CLASS"
done
