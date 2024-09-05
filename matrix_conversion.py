import pandas as pd
import numpy as np

class MatrixOperation:
    def __init__(self):
        self.df = pd.read_excel("AMP_30_03_2020 IMPROVED.xlsx")


    def OneHotEncoder(self, column):
        value_set = set(self.df[column])
        temporal_df = pd.DataFrame()
        for val in value_set:
            temporal_df[f"{column}_{val}"] = np.multiply(self.df[column] == val, 1)
        return temporal_df

    def matrix_conversion(self):
        """
        Conversion of existing AMP database into more machine learning features set
        :return: Dataframe of all needed features
        """
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
            # print(i, col)
            new_df[col] = list(map(lambda x: x.count(code_letters[i]) / len(x), new_df["Sequence"]))
        # print(new_df.round(decimals=3))

        for col in ["Aliphatic", "Aromatic", "NonPolar", "Polar", "Charged", "Basic", "Acidic", ]:
            new_df[col] = np.divide(self.df[col].copy() ,new_df["Length"])

        for col in ['Celularity (Sing, Mult)', 'Tissue (Yes, No)', 'Mesoderm (Yes, NO)', 'Mouthparts(Pro-Deuter)']:
            temp_df = self.OneHotEncoder(col)
            new_df = pd.concat([new_df, temp_df], axis=1)
        return new_df.round(decimals=3)

    def all_existing_labels(self):
        # print(self.df.columns)
        unique = list(set(str(set(self.df.loc[:, "Activity"])).lower().translate({ord(i):None for i in"\'.;}{"}).translate({ord('&'):','}).replace(' ', '').replace("anti-", '').split(sep=',')))
        unique.remove('')
        # print(unique)
        return unique

    def binary_labeling(self):
        # print(str(self.df.loc[0, "Activity"]).capitalize().translate({ord(i):None for i in "\';."}).translate({ord('&'):','}).replace(' ', '').replace("anti-", '').split(sep=','))
        simplification_func = lambda x: (1 if u in x.lower().translate({ord(i): None for i in "\';."}).translate({ord('&'): ','}).replace(' ','').replace("anti-", '').split(sep=',') else 0)
        #df= pd.read_csv('./ML_AMP.csv', sep=',', index_col=0)
        df = pd.DataFrame()
        df["original_labels"] = self.df.loc[:, "Activity"].copy()
        for u in self.all_existing_labels():
            vector = list(map(simplification_func, self.df.loc[:, "Activity"]))
            df[u] = vector
        return df

    def ranking_of_labels(self):
        df = pd.read_csv('./ML_AMP_class.csv', sep=',', index_col=0)
        sums = np.sum(df.iloc[:, 1:], axis=0)
        return sums


MO = MatrixOperation()
# MO.matrix_conversion().to_csv("ML_AMP_features.csv")
# MO.binary_labeling().to_csv('./ML_AMP_class.csv')
# print(MO.ranking_of_labels())