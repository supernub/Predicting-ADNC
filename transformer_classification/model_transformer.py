import torch.nn as nn

class GeneTransformer(nn.Module):
    def __init__(self, num_genes, embedding_dim=128, dim_feedforward=512, head=8, depth=4, dropout=.1):
        super(GeneTransformer, self).__init__()
        self.emb = nn.Embedding(num_genes+1, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.emb.weight, a=-1.0/num_genes, b=1.0/num_genes)
        self.emb.weight.data[0].fill_(0)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=head, 
            batch_first=True, 
            dropout=dropout, 
            dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=depth)

        self.output = nn.Linear(embedding_dim, 34)

    def forward(self, seq, vals):
        x = self.emb(seq)
        x = x * vals.unsqueeze(2)
        x = self.encoder(x)
        x_pooled = x.mean(dim = 1)
        
        return self.output(x_pooled)
