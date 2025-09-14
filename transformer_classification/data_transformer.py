import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

class VariableLengthSequenceDataset(Dataset):
    def __init__(self, X, y): # X is a sparse matrix
        self.seq = []
        self.vals = []
        self.labels = torch.tensor(y)

        print("Initializing dataset...")
        for i in tqdm(range(X.shape[0]), desc="Processing data"):
            x = X[i].toarray()
            x_tensor = torch.from_numpy(x).float()
            
            non_zero_indices = torch.nonzero(x_tensor, as_tuple=True)[1]
            non_zero_indices = non_zero_indices + 1
            self.seq.append(non_zero_indices)
            
            scaling_factor = x_tensor[0][non_zero_indices - 1]
            self.vals.append(scaling_factor)

        # self.seq = nn.utils.rnn.pad_sequence([seq.clone().detach() for seq in self.seq], batch_first=True, padding_value=0)
        # self.vals = nn.utils.rnn.pad_sequence([vals.clone().detach() for vals in self.vals], batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        sequence = self.seq[index]
        scaling_factor = self.vals[index]
        label = self.labels[index]
        return sequence, scaling_factor, label
    
