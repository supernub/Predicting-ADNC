# finetune_geneformer.py
import os, argparse, json, pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    Trainer, TrainingArguments, PreTrainedModel
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# ----- 简单的内存数据集 -----
class PickleTokenDataset(Dataset):
    def __init__(self, pkl_path: str, max_len: int):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        item = self.data[i]
        ids  = item["input_ids"][:self.max_len]
        lab  = item["labels"]
        return {"input_ids": ids, "labels": lab}

# ----- padding collator -----
@dataclass
class Collator:
    pad_id: int = 0
    def __call__(self, features: List[Dict[str, Any]]):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attn_mask = []
        for f in features:
            ids = f["input_ids"]
            pad = [self.pad_id] * (max_len - len(ids))
            input_ids.append(ids + pad)
            attn_mask.append([1]*len(ids) + [0]*len(pad))
            labels.append(f["labels"])
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return batch

# ----- 模型：Geneformer encoder + 头 -----
class OrdinalHead(nn.Module):
    """CORN: 以K-1个阈值进行二分类，推断时统计sigmoid>0.5的个数"""
    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        assert num_classes >= 2
        self.out = nn.Linear(hidden_size, num_classes - 1)
    def forward(self, x):
        return self.out(x)

def corn_loss(logits, targets):
    # logits: (B, K-1), targets: (B,)
    # 构造 y>k 的标签
    B, K_1 = logits.shape
    t = targets.unsqueeze(1).expand(B, K_1)
    ks = torch.arange(K_1, device=logits.device).unsqueeze(0).expand(B, K_1)
    y_gt_k = (t > ks).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, y_gt_k, reduction="mean")

class GeneformerWithHead(PreTrainedModel):
    config_class = AutoConfig
    def __init__(self, config, num_classes: int, loss_type: str = "ce"):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        self.loss_type = loss_type
        hidden = config.hidden_size
        self.num_classes = num_classes
        if loss_type == "ce":
            self.classifier = nn.Linear(hidden, num_classes)
        elif loss_type == "corn":
            self.classifier = OrdinalHead(hidden, num_classes)
        else:
            raise ValueError("loss_type must be ce|corn")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取 pooled（有的模型用 last_hidden_state[:,0] 作为 CLS）
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            if self.loss_type == "ce":
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                loss = corn_loss(logits, labels)
        return {"loss": loss, "logits": logits}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pkl", required=True)
    ap.add_argument("--val_pkl", required=True)
    ap.add_argument("--pretrained", default="ctheodoris/Geneformer")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--loss", choices=["ce","corn"], default="ce")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading config:", args.pretrained)
    config = AutoConfig.from_pretrained(args.pretrained)
    model  = GeneformerWithHead.from_pretrained(
        args.pretrained, config=config,
        num_classes=args.num_classes, loss_type=args.loss
    )

    train_ds = PickleTokenDataset(args.train_pkl, max_len=args.max_len)
    val_ds   = PickleTokenDataset(args.val_pkl,   max_len=args.max_len)
    collator = Collator(pad_id=0)

    steps_per_epoch = len(train_ds) // args.batch_size
    print("Train items:", len(train_ds), "Val items:", len(val_ds), "steps/epoch:", steps_per_epoch)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,                          # fp16
        report_to="none",                   # 可接入 wandb
        dataloader_num_workers=0,           # 先保守
        ddp_find_unused_parameters=False,   # 多卡更稳
    )

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
        logits, labels = eval_pred
        if args.loss == "ce":
            preds = logits.argmax(-1)
        else:
            # corn: 统计有多少阈值通过
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > 0.5).sum(axis=1)
        acc = accuracy_score(labels, preds)
        f1w = f1_score(labels, preds, average="weighted")
        mae = mean_absolute_error(labels, preds)
        return {"accuracy": acc, "f1_weighted": f1w, "mae": mae}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Finished. Best model saved to", args.output_dir)

if __name__ == "__main__":
    main()
