import pandas as pd
from typing import List, Dict
from src.LearningData import LData

class ML_data(LData):

    def features(self, mode:int=3) -> pd.DataFrame: # dodac rozne zbiory
        """
        Method returns features datastet
        :param mode: 1 for only % of aminoacids, 2 %+physicochemical, 3 %+evolutionary data, 4 all, 5 physicochemical+evol
        :return: Specified dataset
        """
        v_sets = [["Aliphatic", "Aromatic", "NonPolar", "Polar", "Charged", "Basic", "Acidic"],
                  ['Kingdom', 'Celularity (Sing, Mult)', 'Tissue (Yes, No)', 'Mesoderm (Yes, NO)', 'Mouthparts(Pro-Deuter)', 'Phyllum', 'Class'],
                  ['Alanine', 'Arginine', 'Asparagine', 'Aspartic_Acid', 'Cysteine', 'Glutamic_Acid', 'Glutamine', 'Glycine', 'Histidine', 'Isoleucine', 'Leucine', 'Lysine', 'Mathionine', 'Phenylalanine', 'Proline', 'Serine', 'Threonine', 'Tryptophan', 'Tyrosine', 'Valine',]]
                #is organism is multi celled   #does it have tissues   #does it make mesoderm and stuff #third options informs if its not natural, maybe its better to not include them?
        temp = []
        for colname in v_sets[1]:
            for column in self.features_df:
                if colname in column:
                    temp.append(column)
        v_sets[1] = temp

        if mode==1:
            return self.features_df.drop(columns=["ID", "Sequence"]+v_sets[0]+v_sets[1]) #only % of aminacids
        elif mode==2:
            return self.features_df.drop(columns=["ID", "Sequence"]+v_sets[1]) #%+physicochemical
        elif mode==3:
            return self.features_df.drop(columns=["ID", "Sequence"]+v_sets[0]) #%+biological
        elif mode==4:
            return self.features_df.drop(columns=["ID", "Sequence"]) #all
        elif mode==5:
            return self.features_df.drop(columns=["ID", "Sequence"]+v_sets[2]) #no %.
        else:
            return "parameter mode: 1 for only % of aminoacids, 2 %+physicochemical, 3 %+evolutionary data, 4 all, 5 without %"


    #cancer cannot harm organisms without tissues, that why I will exclude it
    def __cancer_features(self, mode) -> pd.DataFrame:
        cancer_features_df = self.features_df.loc[(self.features_df['Tissue (Yes, No)_YES'] == 1), :]

        v_sets = [["Aliphatic", "Aromatic", "NonPolar", "Polar", "Charged", "Basic", "Acidic"],
                  ['Kingdom', 'Celularity (Sing, Mult)', 'Tissue (Yes, No)', 'Mesoderm (Yes, NO)',
                   'Mouthparts(Pro-Deuter)', 'Class'],
                  ['Alanine', 'Arginine', 'Asparagine', 'Aspartic_Acid', 'Cysteine', 'Glutamic_Acid', 'Glutamine',
                   'Glycine', 'Histidine', 'Isoleucine', 'Leucine', 'Lysine', 'Mathionine', 'Phenylalanine', 'Proline',
                   'Serine', 'Threonine', 'Tryptophan', 'Tyrosine', 'Valine', ]]
        # is organism is multi celled   #does it have tissues   #does it make mesoderm and stuff #third options informs if its not natural, maybe its better to not include them?
        temp = []
        for colname in v_sets[1]:
            for column in cancer_features_df:
                if colname in column:
                    temp.append(column)
        v_sets[1] = temp

        if mode == 1:
            return cancer_features_df.drop(columns=["ID", "Sequence"] + v_sets[0] + v_sets[1])  # only % of aminacids
        elif mode == 2:
            return cancer_features_df.drop(columns=["ID", "Sequence"] + v_sets[1])  # %+physicochemical
        elif mode == 3:
            return cancer_features_df.drop(columns=["ID", "Sequence"] + v_sets[0])  # %+biological
        elif mode == 4:
            return cancer_features_df.drop(columns=["ID", "Sequence"])  # all
        elif mode == 5:
            return cancer_features_df.drop(columns=["ID", "Sequence"] + v_sets[2])  # no %.
        else:
            raise "parameter mode: 1 for only % of aminoacids, 2 %+physicochemical, 3 %+evolutionary data, 4 all, 5 without %"


    def cancer_df(self, mode) -> Dict[str:pd.DataFrame, str:pd.Series]:
        """

        :param mode: 1 for only % of aminoacids, 2 %+physicochemical, 3 %+evolutionary data, 4 all, 5 physicochemical+evol
        :return: Set of fetures
        """
        cancer_labels = self.labels_df["antitumor"] + self.labels_df["cancercells"] + self.labels_df['anticancer'] > 0
        cancer_features = self.__cancer_features(mode=mode)

        relevant_indexes = cancer_features.index


        return {"features": cancer_features, "labels":cancer_labels.iloc[relevant_indexes]}




