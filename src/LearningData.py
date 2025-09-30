import pandas as pd
from typing import Dict

class LData:
    def __init__(self) -> None:
        self.features_df = pd.read_csv("../ML_AMP_features.csv", sep=',', index_col=0)
        self.labels_df = pd.read_csv(filepath_or_buffer="../ML_AMP_labels.csv", sep=',', index_col=0)
        self.na_indexes = self.features_df[(self.features_df.isna() == True).any(axis=1)].index
        self.drop_na()

    def drop_na(self):
        self.features_df = self.features_df.drop(index=self.na_indexes)
        self.labels_df = self.labels_df.drop(index=self.na_indexes)
        return self

    def labels(self) -> Dict[str, pd.Series]:
        """
        Takes all relevant labels that will be used for analysis
        :keyword:
        antigram+
        antigram-
        antifungal
        anticancer
        antivirial
        :return: List of pandas series of labels
        """
        classes_list = {
            "antigram+": self.labels_df['gram+'],
            "antigram-": self.labels_df['gram-'],
            "antifungal": self.labels_df['antifungal'],
            "anticancer": (self.labels_df["antitumor"] + self.labels_df["cancercells"] + self.labels_df['anticancer'] > 0),
            "antivirial": (self.labels_df["antiviral"] + self.labels_df['hiv'] + self.labels_df['hsv'] > 0),
        }
        return classes_list