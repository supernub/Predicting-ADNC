# train_accelerate.py
import os
import argparse
import math
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error

from dataset_varlen import VariableLengthSequenceDataset, CSRMemmapDataset, pad_collate, load_h5ad_with_split, load_labels_and_split_only
from model_ordinal_transformer import GeneTransformerOrdinal, coral_loss, coral_predict
from typing import List, Dict, Optional, Union, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=str, required=True, help="Path to h5ad")
    p.add_argument("--split_json", type=str, required=True, help="Path to cell_split_A9_80_20.json")
    p.add_argument("--label_col", type=str, default="ADNC_stage", help="obs column name for labels")
    p.add_argument("--label_order", type=str, default="", help="Comma-separated order, e.g. 'Not AD,Low,Intermediate,High' or 'A0,A1,A2,A3'")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--ffn_dim", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs_adnc_ordinal")
    p.add_argument("--eval_every", type=int, default=200)  # steps
    p.add_argument("--class_weight", type=str, default="", help="Comma weights for classes 0..K-1, e.g. '1,1,2,3'")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for CORAL predict")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--memmap_dir", type=str, default="", help="Directory containing CSR memmap arrays; if set, use CSRMemmapDataset")
    return p.parse_args()

def main():
    args = parse_args()

    label_order = [s.strip() for s in args.label_order.split(",")] if args.label_order.strip() else None

    accelerator = Accelerator(mixed_precision="fp16")  # FP16 on 3090
    device = accelerator.device
    accelerator.print(f"Using device: {device}, mixed_precision=fp16")

    # Load data & split
    # === 数据加载（根据是否提供 --memmap_dir 切换实现）===
    if args.memmap_dir and len(args.memmap_dir.strip()) > 0:
        # 只读标签与划分，不再把 h5ad 的 X 读进内存
        y, train_idx, val_idx, num_genes, num_classes, mapping = load_labels_and_split_only(
            args.h5ad, args.split_json, args.label_col, label_order
        )
        train_ds = CSRMemmapDataset(args.memmap_dir, y, train_idx)
        val_ds   = CSRMemmapDataset(args.memmap_dir, y, val_idx)
    else:
        # 常规路径（无 memmap）：读取 X，再用“惰性”Dataset
        X, y, train_idx, val_idx, num_genes, num_classes, mapping = load_h5ad_with_split(
            args.h5ad, args.split_json, args.label_col, label_order
        )
        train_ds = VariableLengthSequenceDataset(X, y, train_idx)
        val_ds   = VariableLengthSequenceDataset(X, y, val_idx)
    # === 数据加载结束 ===

    collate = lambda batch: pad_collate(batch, pad_idx=0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    model = GeneTransformerOrdinal(
        num_genes=num_genes,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        dim_feedforward=args.ffn_dim,
        nhead=args.nhead,
        depth=args.depth,
        dropout=args.dropout,
        pad_idx=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optional class weights
    if args.class_weight.strip():
        w_list = [float(x) for x in args.class_weight.split(",")]
        assert len(w_list) == num_classes, "class_weight length must equal num_classes"
        class_weights = torch.tensor(w_list, dtype=torch.float32, device=device)
    else:
        class_weights = None

    model, opt, train_dl, val_dl = accelerator.prepare(model, opt, train_dl, val_dl)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, (seqs, vals, labels, attn_mask) in enumerate(train_dl, start=1):
            logits = model(seqs, vals, attn_mask=attn_mask)
            loss = coral_loss(logits, labels, reduction="mean",
                              class_weights=class_weights)

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            running += loss.item()
            global_step += 1

            if global_step % args.eval_every == 0:
                accelerator.print(f"[epoch {epoch} step {global_step}] train_loss={running/args.eval_every:.4f}")
                running = 0.0
                # eval
                val_loss, val_acc, val_f1, val_mae = evaluate(accelerator, model, val_dl, class_weights, args.threshold)
                accelerator.print(f"  VAL: loss={val_loss:.4f} acc={val_acc:.4f} f1w={val_f1:.4f} mae={val_mae:.4f}")

                if accelerator.is_main_process and val_mae < best_val_mae:
                    best_val_mae = val_mae
                    ckpt = os.path.join(args.output_dir, f"best_epoch{epoch}_step{global_step}_mae{val_mae:.4f}.pt")
                    accelerator.print(f"  >> saving {ckpt}")
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({
                        "model": unwrapped.state_dict(),
                        "num_genes": num_genes,
                        "num_classes": num_classes,
                        "mapping": mapping,
                        "hparams": vars(args)
                    }, ckpt)

        # epoch-end eval
        val_loss, val_acc, val_f1, val_mae = evaluate(accelerator, model, val_dl, class_weights, args.threshold)
        accelerator.print(f"[epoch {epoch} END] VAL: loss={val_loss:.4f} acc={val_acc:.4f} f1w={val_f1:.4f} mae={val_mae:.4f}")
        if accelerator.is_main_process and val_mae < best_val_mae:
            best_val_mae = val_mae
            ckpt = os.path.join(args.output_dir, f"best_epoch{epoch}_end_mae{val_mae:.4f}.pt")
            accelerator.print(f"  >> saving {ckpt}")
            unwrapped = accelerator.unwrap_model(model)
            torch.save({
                "model": unwrapped.state_dict(),
                "num_genes": num_genes,
                "num_classes": num_classes,
                "mapping": mapping,
                "hparams": vars(args)
            }, ckpt)

def evaluate(accelerator, model, dataloader, class_weights, threshold: float):
    model.eval()
    total_loss = 0.0
    ys, preds = [], []

    with torch.no_grad():
        for seqs, vals, labels, attn_mask in dataloader:
            logits = model(seqs, vals, attn_mask=attn_mask)
            loss = coral_loss(logits, labels, reduction="mean", class_weights=class_weights)
            total_loss += loss.item() * labels.size(0)

            yhat = coral_predict(logits, threshold=threshold)
            ys.append(labels.cpu().numpy())
            preds.append(yhat.cpu().numpy())

    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    avg_loss = total_loss / len(ys)
    acc = accuracy_score(ys, preds)
    f1w = f1_score(ys, preds, average="weighted")
    mae = mean_absolute_error(ys, preds)  # 序数任务很关键的指标
    model.train()
    return avg_loss, acc, f1w, mae

if __name__ == "__main__":
    main()
