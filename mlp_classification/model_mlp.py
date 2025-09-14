import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc
from sklearn.metrics import f1_score
import hdf5plugin
import wandb
import argparse
import os

wandb.login(key="a0e3bae070669cbc36f3f33105a9e681c889b00f")
wandb.init(project="gene-interact-MLP")
print(f"Wandb run initialized as {wandb.run.name}")

argparser = argparse.ArgumentParser()
argparser.add_argument("--cell_type", type=str, required=True)

args = argparser.parse_args()
cell_type = args.cell_type

def load_data(data_path):
    print(f"Loading data from {data_path}")
    data_cell = sc.read_h5ad(data_path)

    filtered_data = data_cell[data_cell.obs['Overall AD neuropathological Change'].isin(['Not AD', 'High'])]
    X_filtered = filtered_data.X
    y_filtered = filtered_data.obs['Overall AD neuropathological Change']
    label_mapping = {'Not AD': 0, 'High': 1}
    y_encoded = y_filtered.map(label_mapping)
        
    return list(zip(X_filtered.todense(), y_encoded))


# train_dataloader = DataLoader(load_data(f"data/split_train/neuron_data.h5ad"), batch_size=128,  shuffle = True)
# test_dataloader = DataLoader(load_data(f"data/split_test/neuron_data.h5ad"), batch_size=128,  shuffle = False)
train_dataloader = DataLoader(load_data(f"data/split_train/cell_type_{cell_type}.h5ad"), batch_size=128,  shuffle = True)
test_dataloader = DataLoader(load_data(f"data/split_test/cell_type_{cell_type}.h5ad"), batch_size=128,  shuffle = False)

class MLP(nn.Module):
    def __init__(self, num_genes, hidden1=128, hidden2=64):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_genes, hidden1, bias=False),
            nn.Dropout(p=0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden1, hidden2),
            nn.Dropout(p=0.5),
            nn.Softplus(beta=1, threshold=20),
            nn.Linear(hidden2, 2)  # Two output neurons for binary classification
        )

    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        out = self.layers(x)
        return out

num_genes = 36601
model = MLP(num_genes)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = nn.DataParallel(model, device_ids=[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Training loop
def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (batch_sequences, batch_labels) in enumerate(dataloader):
        # Move tensors to GPU
        batch_sequences = batch_sequences.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_sequences)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} Total Loss: {total_loss / (batch_idx+1):.4f}")
            wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": total_loss / (batch_idx+1)})

    return total_loss / len(dataloader)

# Evaluation loop
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_labels) in enumerate(dataloader):
            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_sequences)
            loss = criterion(output, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = (100 * (correct / total))
    f1_value = f1_score(y_true, y_pred, average="weighted") * 100
    
    print(f"Test loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}, f1_score: {f1_value}")
    
    return total_loss / len(dataloader), accuracy, f1_value

os.makedirs(f"output_models/MLP/{cell_type}/", exist_ok=True)

# Training and evaluation
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs}:")
    train_loss = train(model, train_dataloader, optimizer, criterion, epoch)
    test_loss, acc, f1_value = evaluate(model, test_dataloader, criterion)
    wandb.log({"epoch": epoch, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
    # save model
    if epoch > 0 and (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), f"output_models/MLP/{cell_type}/mlp_epoch_{epoch}.pth")