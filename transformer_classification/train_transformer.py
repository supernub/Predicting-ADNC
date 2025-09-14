import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import scanpy as sc
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
import os
import json
import argparse
import hdf5plugin

from model_transformer import GeneTransformer
from dataset_transformer import VariableLengthSequenceDataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def init(JSON_PATH):
    wandb.login(key="24b9dd36f6f7c433e1184bf8940921331fcf0957")
    wandb.init(project="gene-interact")
    print(f"Wandb run initialized as {wandb.run.name}")

    # hyperparameters
    config = json.load(open(JSON_PATH))
    wandb.config.update({
        "num_genes": config["num_genes"],
        "embedding_dim": config["embedding_dim"],
        "dim_feedforward": config["dim_feedforward"],
        "head": config["head"],
        "depth": config["depth"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"]
    })
    return config

# def load_data_single_piece(data_path, is_test, use_split=False, train_path=None, test_path=None):
#     # Otherwise, load the merged file and split it
#     print(f"Loading data from {data_path}...")
#     data_cell = sc.read_h5ad(data_path)
    
#     # Use celltype_id as labels directly
#     X = data_cell.X
#     y = data_cell.obs['celltype_id'].astype(int).values
    
#     print(f"Data shape: {X.shape}, Number of classes: {len(np.unique(y))}")
    
#     # Split into train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     if not is_test:
#         train_dataset = VariableLengthSequenceDataset(X_train, y_train)
#     else:
#         train_dataset = None
        
#     test_dataset = VariableLengthSequenceDataset(X_test, y_test)
    
#     return train_dataset, test_dataset

def load_data(train_path=None, test_path=None):
    # Load test data
    print(f"Loading test data from {test_path}...")
    test_data = sc.read_h5ad(test_path)
    
    # Get test features and labels
    X_test = test_data.X
    y_test = test_data.obs['celltype_id'].astype(int).values
    
    print(f"Test data shape: {X_test.shape}, Number of classes: {len(np.unique(y_test))}")
    test_dataset = VariableLengthSequenceDataset(X_test, y_test)
    
    # Load train data
    print(f"Loading train data from {train_path}...")
    train_data = sc.read_h5ad(train_path)
    
    # Get train features and labels
    X_train = train_data.X
    y_train = train_data.obs['celltype_id'].astype(int).values
    
    print(f"Train data shape: {X_train.shape}, Number of classes: {len(np.unique(y_train))}")
    train_dataset = VariableLengthSequenceDataset(X_train, y_train)
    
    return train_dataset, test_dataset


def train(train_dataset, test_dataset, config, args):
    def collate_fn(batch):
        sequences, scaling_factors, labels = zip(*batch)
        # only fill current batch
        sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
        scaling_factors_padded = nn.utils.rnn.pad_sequence(scaling_factors, batch_first=True, padding_value=0)
        
        labels = torch.tensor(labels)
        
        return sequences_padded, scaling_factors_padded, labels

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle = True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle = False, collate_fn=collate_fn)
    
    model = GeneTransformer(config["num_genes"], config["embedding_dim"], config["dim_feedforward"], config["head"], config["depth"], config["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    model = nn.DataParallel(model, device_ids=args.device_ids)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_f1 = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        test_loss, acc, f1_value = 0, 0, 0

        for batch_idx, (batch_sequences, batch_values, batch_labels) in enumerate(train_dataloader): #enumerate(dataloader)
            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_sequences, batch_values)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % args.train_loss_log == 0:
                print(f"    Batch {batch_idx}/{len(train_dataloader)} Total Loss: {total_loss / (batch_idx+1):.4f}")
                wandb.log({"epoch": epoch, "step": batch_idx, "train_loss": total_loss / (batch_idx+1)})
                
            if (batch_idx % args.eval_every == 0) and batch_idx > 0:
                print("start testing")
                test_loss, acc, f1_value = test(model, test_dataloader, criterion, args.device)
                print("test_loss: ", test_loss, "test_acc: ", acc," f1_value: ", f1_value)
                wandb.log({"epoch": epoch, "step": batch_idx, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
                if f1_value > best_f1:
                    best_f1 = f1_value
                    print(f"    Model saved to {args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                    torch.save(model.state_dict(), f"{args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
        
        if args.epoch_wise_test:
            test_loss, acc, f1_value = test(model, test_dataloader, criterion, args.device)
            print({"epoch": epoch, "test_loss": test_loss, "accuracy": acc, "f1_score": f1_value})
            if f1_value > best_f1:
                best_f1 = f1_value
                print(f"    Model saved to {args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                torch.save(model.state_dict(), f"{args.output}/{wandb.run.name}/epoch_{epoch}_batch_{batch_idx}.pth")
                    
    return


def test(model, test_dataloader, criterion, device):
    print("     Evaluating...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_values, batch_labels) in enumerate(test_dataloader):

            # Move tensors to GPU
            batch_sequences = batch_sequences.to(device)
            batch_values = batch_values.to(device)
            batch_labels = batch_labels.to(device)

            output = model(batch_sequences, batch_values)
            loss = criterion(output, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            y_true.extend(batch_labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = (100 * (correct / total))
    f1_value = f1_score(y_true, y_pred, average="weighted") * 100
    
    print(f"Test loss: {total_loss / len(test_dataloader)}, Accuracy: {accuracy}, f1_score: {f1_value}")
    
    return total_loss / len(test_dataloader), accuracy, f1_value
    
def parse_list(arg_value):
    return [int(item) for item in arg_value.split(',')]

def load_model(model_path, config):
    # original saved file with DataParallel
    state_dict = torch.load(model_path)

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    # load params
    model = GeneTransformer(config["num_genes"], config["embedding_dim"], config["dim_feedforward"], config["head"], config["depth"], config["dropout"])
    model.load_state_dict(new_state_dict)
    
    return model
    
    
if __name__ == "__main__":
    ## example usage:
    ## python transformer/main.py --config config.json --data SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad --output output_models --device_ids 0,1,2,3 --eval_every 1000 --epochs 4 --device cuda --model_path model.pth
    parser = argparse.ArgumentParser(description="gene-transformer")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the JSON file containing the hyperparameters")
    parser.add_argument("--data", type=str, default="data/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad", help="Path to the h5ad file containing the data")
    parser.add_argument("--output", type=str, default="output_models/global_training", help="Path to the output directory")
    parser.add_argument("--train_path", type=str, default="output_models/global_training", help="Path to the train data")
    parser.add_argument("--test_path", type=str, default="output_models/global_training", help="Path to the test data")
    parser.add_argument("--device_ids", type=parse_list, default=[0, 1, 2, 3], help="List of device ids to use")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate the model every n steps")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train the model")
    parser.add_argument("--is_test", action="store_true", help="Test the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and evaluation")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the model")
    parser.add_argument("--epoch_wise_test", type=bool, default=False, help="Test the model after each epoch")
    parser.add_argument("--train_loss_log", type=int, default=50, help="Log the train loss every n steps")
    args = parser.parse_args()
    
    setup_seed(42)
    
    if (not args.is_test):
        config = init(args.config)
        os.makedirs(f"{args.output}/{wandb.run.name}", exist_ok=True)
    train_dataset, test_dataset = load_data(args.train_path, args.test_path)
    
    if (not args.is_test):
        train(train_dataset, test_dataset, config, args)
    else:
        config = json.load(open(args.config))
        test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle = False)
        model = load_model(args.model_path, config)
        model = nn.DataParallel(model, device_ids=args.device_ids)
        model = model.to(args.device)
        test(model, test_dataloader, nn.CrossEntropyLoss(), args.device)