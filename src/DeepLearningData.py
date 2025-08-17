import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from src.LearningData import LData

class DLDataSplit(LData):
    def __init__(self,):
        super().__init__()


    def __splitting(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.1, random_state=42)

        return X_train, y_train, X_test, y_test

    def __getitem__(self, label):
        X_train_whole, y_train_whole, X_test, y_test = self.__splitting(self.__features_df(), self.labels()[label])
        test_data = {'X':X_test, 'y':y_test}
        X_train, y_train, X_val, y_val = self.__splitting(X_train_whole, y_train_whole)
        validation_data = {'X':X_val, 'y':y_val}
        train_data = {"X":X_train, "y":y_train}
        return train_data, validation_data, test_data


class DL_data(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.Series) -> None:
        super().__init__()
        self.aminoacids = ["N",'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G',
                           'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T',
                           'W', 'Y','V', 'X'
                           ]
        self._amino_acid_binary = self._binary_encoded_dictionary()
        self.seqs = features["Sequence"]
        self.seq_len = np.max(features["Length"])
        self.labels = torch.tensor(labels).unsqueeze(1)

        self.one_hot_seqs = self._one_hot_seqs()


    def _binary_encoded_dictionary(self) -> Dict[str, List[int]]:
        try:
            encoded_dictionary = pd.read_csv("../cache/encoded_dictionary.csv", index_col=0)

        except FileNotFoundError:

            empty_matrix = np.zeros([len(self.aminoacids), len(self.aminoacids)])
            encoded_dictionary = {}
            i = 0
            while i < len(self.aminoacids):
                if i==0:
                    encoded_dictionary[self.aminoacids[i]] = empty_matrix[i]
                else:
                    aminoacid_bin = empty_matrix[i]
                    aminoacid_bin[i - 1] = 1
                    encoded_dictionary[self.aminoacids[i]] = aminoacid_bin
                    i += 1
            df = pd.DataFrame(encoded_dictionary)
            df.to_csv("../cache/encoded_dictionary.csv")
        return encoded_dictionary

    def _one_hot_seqs(self):
        stacks = torch.stack(list(map(lambda x: torch.tensor(self.one_hot_encode_sequence(x)), self.seqs)))
        return stacks


    def one_hot_encode_sequence(self, sequence: str) -> List[List[int]]:
        encoded_sequence =[]# list(map(lambda x: self._amino_acid_binary[x], sequence))
        i = 0
        while i<self.seq_len:
            try:
                encoded_sequence.append(self._amino_acid_binary[sequence[i]])
            except IndexError:
                encoded_sequence.append(self._amino_acid_binary["N"])
            i+=1
        return encoded_sequence

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        features = self.one_hot_seqs
        labels = self.labels
        return features, labels



# def DeepLearningData(train_df, test_df, batch)
#
#
#
# # seqs = ["ARN", "DCE"]
# # print(torch.stack(list(map(lambda x: torch.tensor(DL.one_hot_encode_sequence(x)), seqs))))