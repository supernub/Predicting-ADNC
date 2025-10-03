#!/bin/bash
set -e

DATA="../training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad"
SPLIT="split_donors.json"
OUTDIR="mil_a9_runs_v918"

classes=("Neuronal: Glutamatergic" "Neuronal: GABAergic" "Non-neuronal and Non-neural")

mkdir -p "$OUTDIR"

for CLS in "${classes[@]}"; do
  SAFE=$(echo "$CLS" | tr ' /:' '__')
  echo "=== Running Class = $CLS ==="
    python mlp_donor_adnc_ver918.py \
    --data-path "$DATA" \
    --split-json "$SPLIT" \
    --donor-col "Donor ID" \
    --adnc-col "ADNC" \
    --filter-col "Class" \
    --filter-value "$CLS" \
    --icontains \
    --pool mean \
    --batch-donors 12 \
    --k-per-donor-train 2048 \
    --k-per-donor-eval 6000 \
    --epochs 40 \
    --patience 10 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --out-dir "$OUTDIR/$SAFE" \
    --save-model \
    --save-metrics
done
