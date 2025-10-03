# model_ordinal_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple


class GeneTransformerOrdinal(nn.Module):
    """
    Encoder: Embedding -> scale by expression -> TransformerEncoder -> mean pool
    Head: CORAL ordinal head (shared linear + K-1 biases)
    """
    def __init__(
        self,
        num_genes: int,
        num_classes: int,
        embedding_dim: int = 128,
        dim_feedforward: int = 512,
        nhead: int = 8,
        depth: int = 4,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.num_classes = num_classes

        # token 0 reserved for PAD; genes indexed from 1..num_genes
        self.emb = nn.Embedding(num_genes + 1, embedding_dim, padding_idx=pad_idx)
        nn.init.uniform_(self.emb.weight, a=-1.0/num_genes, b=1.0/num_genes)
        with torch.no_grad():
            self.emb.weight.data[pad_idx].fill_(0.0)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # CORAL head: shared weight vector w \in R^d  +  (K-1) thresholds/biases b_k
        self.shared_fc = nn.Linear(embedding_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        nn.init.xavier_uniform_(self.shared_fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, vals, attn_mask=None):
        """
        seq: LongTensor [B, L] with PAD=0, gene ids start from 1
        vals: FloatTensor [B, L] expression values aligned with seq
        attn_mask: BoolTensor [B, L] True for PAD positions to be masked out
        """
        x = self.emb(seq)                       # [B, L, D]
        x = x * vals.unsqueeze(-1)              # scale by expression
        x = self.encoder(x, src_key_padding_mask=attn_mask)  # attn_mask: True=pad
        x = x.masked_fill(attn_mask.unsqueeze(-1), 0.0) if attn_mask is not None else x
        # mean pool over non-pad tokens
        if attn_mask is not None:
            valid_count = (~attn_mask).sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
            x_pooled = x.sum(dim=1) / valid_count
        else:
            x_pooled = x.mean(dim=1)

        x_pooled = self.dropout(x_pooled)

        # shared logits per threshold: shape [B, 1] then add (K-1) biases → [B, K-1]
        base = self.shared_fc(x_pooled)               # [B, 1]
        logits = base + self.biases.view(1, -1)       # [B, K-1]
        return logits  # CORAL logits for P(y > k)

def coral_loss(logits, targets, reduction="mean", class_weights=None):
    """
    logits: [B, K-1] (no sigmoid)
    targets: [B] with values in {0,1,...,K-1}
    CORAL loss = sum_k BCEWithLogits(sigmoid(logit_k), 1_{y > k})
    """
    B, Km1 = logits.shape
    # Build binary targets for each threshold: target_k = 1 if y > k else 0
    thresholds = torch.arange(Km1, device=targets.device).view(1, -1)  # [1, K-1]
    bin_targets = (targets.view(-1, 1) > thresholds).float()           # [B, K-1]

    if class_weights is not None:
        # Optional: weight per class index (0..K-1). Map to per-example weights.
        w = class_weights[targets].view(-1, 1)  # [B,1]
        loss = F.binary_cross_entropy_with_logits(logits, bin_targets, weight=w, reduction="none")
    else:
        loss = F.binary_cross_entropy_with_logits(logits, bin_targets, reduction="none")

    # sum over thresholds
    loss = loss.sum(dim=1)  # [B]
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss  # [B]

@torch.no_grad()
def coral_predict(logits, threshold=0.5):
    """
    logits: [B, K-1]
    Predict class = count(sigmoid(logit_k) > threshold)
    returns LongTensor [B]
    """
    probs = torch.sigmoid(logits)           # [B, K-1]
    preds = (probs > threshold).sum(dim=1)  # in {0..K-1}
    return preds
