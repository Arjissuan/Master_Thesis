import pandas as pd
import numpy as np

class MatrixOperation:
    def __init__(self):
        self.df = pd.read_excel("AMP_30_03_2020 IMPROVED.xlsx")

    def matrix_conversion(self):
        new_df = pd.DataFrame(
            columns=["ID", "Sequence", "Length", "Alanine", "Arginine", "Asparagine", "Aspartic_Acid", "Cysteine",
                     "Glutamic_Acid", "Glutamine", "Glycine", "Histidine", "Isoleucine", "Leucine",
                     "Lysine", "Mathionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan",
                     "Tyrosine", "Valine"])

        code_letters = (
        'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

        new_df["Sequence"] = self.df["Sequence"].copy()
        new_df["ID"] = self.df["ID"].copy()
        new_df["Length"] = list(map(lambda x: len(x), new_df["Sequence"]))
        for i, col in enumerate(new_df.columns[3:]):
            print(i, col)
            new_df[col] = list(map(lambda x: x.count(code_letters[i]) / len(x), new_df["Sequence"]))
        print(new_df.round(decimals=3))

        return new_df.round(decimals=3)

    def class_generation(self):
        # print(self.df.columns)
        unique = list(set(str(set(self.df.loc[:, "Activity"])).capitalize().translate({ord(i):None for i in"\'}{;."}).translate({ord('&'):','}).replace(' ', '').replace("anti-", '').split(sep=',')))
        unique.remove('')
        # unique.remove("antifungalnn")
        print(unique)
        # uni = []
        # cnt = []
        # for cell in self.df.loc[:, "Activity"]:
        #     for atribute in cell.split(','):
        #
        # print(uni)



MO = MatrixOperation()
# MO.matrix_conversion().to_csv("ML_AMP.csv")
MO.class_generation()


