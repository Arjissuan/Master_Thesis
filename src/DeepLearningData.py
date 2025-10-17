import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import torch.nn.functional as F
from src.LearningData import LData


class DL_data(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.Series, max_len: int = None) -> None:
        super().__init__()
        self.aminoacids = [
            'A','R','N','D','C','E','Q','G','H','I',
            'L','K','M','F','P','S','T','W','Y','V','X','-'
        ]
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aminoacids)}
        self.num_tokens = len(self.aminoacids)

        self.seqs = features["Sequence"].tolist()
        self.seq_len = max_len if max_len else int(np.max(features["Length"]))

        # single integer class label per sample
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def one_hot_encode_sequence(self, sequence: str) -> torch.Tensor:
        sequence = sequence.ljust(self.seq_len, '-')[:self.seq_len]
        idxs = [self.aa_to_idx.get(aa, self.aa_to_idx['X']) for aa in sequence]
        idxs = torch.tensor(idxs, dtype=torch.long)
        return F.one_hot(idxs, num_classes=self.num_tokens).float()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        one_hot_seq = self.one_hot_encode_sequence(self.seqs[idx])
        one_hot_seq = one_hot_seq.unsqueeze(0)  # (1, seq_len, vocab_size)
        label = self.labels[idx]                # scalar int
        return one_hot_seq, label

class DLDataSplit:
    def __init__(self, X, Y, batch_size=32, max_len=None):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.max_len = max_len

    def __splitting(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

        return X_train, y_train, X_test, y_test

    def __getitem__(self):
        X_train, X_test, y_train, y_test = self.__splitting(self.X, self.Y)
        X_train, X_val, y_train, y_val = self.__splitting(X_train, y_train)

        # create datasets
        train_ds = DL_data(X_train, y_train, max_len=self.max_len)
        val_ds = DL_data(X_val, y_val, max_len=self.max_len)
        test_ds = DL_data(X_test, y_test, max_len=self.max_len)

        # dataloaders
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

        #
        # X_train_whole, y_train_whole, X_test, y_test = self.__splitting(self.X, self.Y)
        # test_data = {'X':X_test, 'y':y_test}
        # X_train, y_train, X_val, y_val = self.__splitting(X_train_whole, y_train_whole)
        # validation_data = {'X':X_val, 'y':y_val}
        # train_data = {"X":X_train, "y":y_train}
        # return train_data, validation_data, test_data



# seqs = ["ARN", "DCE"]
# DL = DL_data()
# print(torch.stack(list(map(lambda x: torch.tensor(DL.one_hot_encode_sequence(x)), seqs))))