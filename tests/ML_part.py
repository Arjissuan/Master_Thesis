from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
import os

"""
0. W metodzie features() klasy ML_data() dodac finckjonalnosc pozwalajaca na wybor zbioru cech ktore wykorzystane zostana do ML
1. Dodac dodatkowe cechy do zbioru danych, 4 grupy cech. 
    1. Czestotliwosc wystepowania aminokwasow(to juz istnieje)
    2. 1+cechy na zielono
    3. 1+wlasciwosci fizyko chemiczne
    4. wszystko z poprzdnich
2. Zmodyfikowac architekture siecii neuronwej i przeprowadzic testy. 6 warstw, 30 wejsc->64->16->8->4->2->1
2. Crossvalidacja (1/2 zrobione) 
3. Selekcja cech?
"""

# features_df = pd.read_csv("../ML_AMP_features.csv", sep=',', index_col=0)
# na_indexes = features_df[(features_df.isna() == True).any(axis=1)].index
# features_df = features_df.drop(index=na_indexes)
# ids = features_df["ID"].copy()
# features = features_df.drop(columns=["ID", "Sequence"])
#
# class_df = pd.read_csv(filepath_or_buffer="../ML_AMP_class.csv", sep=',', index_col=0)
# class_df = class_df.drop(index=na_indexes)
# gramp = class_df["gram+"]
# gramm = class_df["gram-"]
# fung = class_df["antifungal"]
# cancer = (class_df["antitumor"] + class_df["cancercells"] + class_df['anticancer'] > 0)
# vir = (class_df["antiviral"] + class_df['hiv'] + class_df['hsv'] > 0)

class ML_data:
    def __init__(self):
        self.__features_df = pd.read_csv("../ML_AMP_features.csv", sep=',', index_col=0)
        self.__class_df = pd.read_csv(filepath_or_buffer="../ML_AMP_class.csv", sep=',', index_col=0)
        self.__na_indexes = self.__features_df[(self.__features_df.isna() == True).any(axis=1)].index

    def drop_na(self):
        self.__features_df = self.__features_df.drop(index=self.__na_indexes)
        self.__class_df = self.__class_df.drop(index=self.__na_indexes)

    def features(self, mode:int) -> pd.DataFrame: # dodac rozne zbiory
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

mld = ML_data()
mld.drop_na()

def data_split(clas_vect, mode:int):
    X_train, X_test, y_train, y_test = train_test_split(mld.features(mode=mode), clas_vect, test_size=0.3, random_state=14)
    return X_train, X_test, y_train, y_test

def classication(clss:list, mode:int):
    acc_klas = []
    for klas in clss:
        # data split
        X_train, X_test, y_train, y_test = data_split(klas, mode)
        # learning
        svc = SVC(kernel="rbf")
        forest = RandomForestClassifier()
        bayes = GaussianNB()
        mlp = MLPClassifier()

        svc.fit(X_train, y_train)
        forest.fit(X_train, y_train)
        bayes.fit(X_train, y_train)
        mlp.fit(X_train, y_train)

        #testing
        svc_pred = svc.predict(X_test), y_test
        forest_pred = forest.predict(X_test), y_test
        bayes_pred = bayes.predict(X_test), y_test
        mlp_pred = mlp.predict(X_test), y_test

        acc_klas.append([{"type":"SVC",
                         "acc":accuracy_score(svc_pred[1], svc_pred[0]),
                         "f1":f1_score(svc_pred[1], svc_pred[0]),
                         "recall":recall_score(svc_pred[1], svc_pred[0]),
                         "precision":precision_score(svc_pred[1], svc_pred[0])
                         },
                        {"type": "Random forest",
                         "acc": accuracy_score(forest_pred[1], forest_pred[0]),
                         "f1": f1_score(forest_pred[1], forest_pred[0]),
                         "recall": recall_score(forest_pred[1], forest_pred[0]),
                         "precision": precision_score(forest_pred[1], forest_pred[0])
                         },
                         {"type": "Bayes",
                          "acc": accuracy_score(bayes_pred[1], bayes_pred[0]),
                          "f1": f1_score(bayes_pred[1], bayes_pred[0]),
                          "recall": recall_score(bayes_pred[1], bayes_pred[0]),
                          "precision": precision_score(bayes_pred[1], bayes_pred[0])
                          },
                         {"type": "MLP",
                          "acc": accuracy_score(mlp_pred[1], mlp_pred[0]),
                          "f1": f1_score(mlp_pred[1], mlp_pred[0]),
                          "recall": recall_score(mlp_pred[1], mlp_pred[0]),
                          "precision": precision_score(mlp_pred[1], mlp_pred[0])
                          }
                         ]
                        )
    return acc_klas


for j in (1,2,3,4):
    statistics = classication(list(map(lambda x: x, mld.classes())), mode=j)
    class_names = ['gram_plus', 'gram_minus', 'fungi', 'cancer', 'viruses']
    for i, dt in enumerate(statistics):
        df = pd.DataFrame(dt)
        df.to_csv(f"./{j}/{class_names[i]}.csv", sep=',')

