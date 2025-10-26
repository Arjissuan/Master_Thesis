import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


class DLData(Dataset):
    """Dataset for peptide sequences with one-hot encoding."""

    AMINOACIDS = [
        'A','R','N','D','C','E','Q','G','H','I',
        'L','K','M','F','P','S','T','W','Y','V','X','-'
    ]
    AA_TO_IDX = {aa: i for i, aa in enumerate(AMINOACIDS)}
    NUM_TOKENS = len(AMINOACIDS)

    def __init__(self, features: pd.DataFrame, labels: pd.Series, max_len: int = None):
        super().__init__()
        self.seqs = features["Sequence"].tolist()
        self.seq_len = max_len if max_len else int(np.max(features["Length"]))
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def one_hot_encode_sequence(self, sequence: str) -> torch.Tensor:
        """Pad sequence to max length and return one-hot encoding."""
        sequence = sequence.ljust(self.seq_len, '-')[:self.seq_len]
        idxs = [self.AA_TO_IDX.get(aa, self.AA_TO_IDX['X']) for aa in sequence]
        idxs = torch.tensor(idxs, dtype=torch.long)
        return F.one_hot(idxs, num_classes=self.NUM_TOKENS).float()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        one_hot_seq = self.one_hot_encode_sequence(self.seqs[idx])
        one_hot_seq = one_hot_seq.unsqueeze(0)  # (1, seq_len, vocab_size)
        label = self.labels[idx]                # scalar int
        return one_hot_seq, label


class DLDataSplit:
    """Handles train/val/test split and DataLoader creation."""

    def __init__(self, X: pd.DataFrame, Y: pd.Series, batch_size: int = 32, max_len: int = None):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.max_len = max_len

    def _split(self, X, y, test_size=0.1):
        return train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    def get_loaders(self):
        """Split dataset and return train, val, and test DataLoaders."""
        X_train, X_test, y_train, y_test = self._split(self.X, self.Y, test_size=0.1)
        X_train, X_val, y_train, y_val = self._split(X_train, y_train, test_size=0.1)

        train_ds = DLData(X_train, y_train, max_len=self.max_len)
        val_ds = DLData(X_val, y_val, max_len=self.max_len)
        test_ds = DLData(X_test, y_test, max_len=self.max_len)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
