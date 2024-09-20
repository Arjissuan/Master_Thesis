import pandas as pd

class ML_data:
    def __init__(self):
        self.__features_df = pd.read_csv("../ML_AMP_features.csv", sep=',', index_col=0)
        self.__class_df = pd.read_csv(filepath_or_buffer="../ML_AMP_class.csv", sep=',', index_col=0)
        self.__na_indexes = self.__features_df[(self.__features_df.isna() == True).any(axis=1)].index

    def drop_na(self):
        self.__features_df = self.__features_df.drop(index=self.__na_indexes)
        self.__class_df = self.__class_df.drop(index=self.__na_indexes)
        return self

    def features(self, mode:int=4) -> pd.DataFrame: # dodac rozne zbiory
        v_sets = [["Aliphatic", "Aromatic", "NonPolar", "Polar", "Charged", "Basic", "Acidic"],
                  ['Celularity (Sing, Mult)', 'Tissue (Yes, No)', 'Mesoderm (Yes, NO)', 'Mouthparts(Pro-Deuter)']]
        temp = []
        for colname in v_sets[1]:
            for column in self.__features_df:
                if colname in column:
                    temp.append(column)
        v_sets[1] = temp
        if mode==1:
            return self.__features_df.drop(columns=["ID", "Sequence"]+v_sets[0]+v_sets[1])
        elif mode==2:
            return self.__features_df.drop(columns=["ID", "Sequence"]+v_sets[1])
        elif mode==3:
            return self.__features_df.drop(columns=["ID", "Sequence"]+v_sets[0])
        elif mode==4:
            return self.__features_df.drop(columns=["ID", "Sequence"])

    def classes(self):
        classes_list = [self.__class_df['gram+'],
                        self.__class_df['gram-'],
                        self.__class_df['antifungal'],
                        (self.__class_df["antitumor"] + self.__class_df["cancercells"] + self.__class_df['anticancer'] > 0),
                        (self.__class_df["antiviral"] + self.__class_df['hiv'] + self.__class_df['hsv'] > 0)
                        ]
        return classes_list