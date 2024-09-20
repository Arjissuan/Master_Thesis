import pandas as pd
import numpy as np
from src.MachineLearningData import ML_data
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class KFold(ML_data):
    def __init__(self, n=10, t=0.7, r=42, m=1):
        super().__init__()
        self.drop_na()
        self.cross_val_data = list(map(lambda x:self.data_split(n, t, r, m, x), self.classes()))
        self.__mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), solver='adam', activation='tanh')
        self.__rfc = RandomForestClassifier(n_estimators=1000)
        self.__svc = SVC()
        self.__bayes = GaussianNB()

    def data_split(self, n:int, t:float, r:int, m:int, clss:pd.Series):
        """

        :param n: number of splits
        :param t: test size
        :param r: random state
        :param m: features mode look into, mode are sets of features, there are 4 sets numbered 1 to 4
        :param clss: which class used for classification
        :return: splitted data ready for cross-validaiton
        """
        sss = StratifiedShuffleSplit(n_splits=n, train_size=t, random_state=r)
        data_split = []
        for train_indx, test_indx in sss.split(self.features(mode=m), clss):
            test_x, test_y = self.features(mode=m).iloc[test_indx, :], clss.iloc[test_indx]
            train_x, train_y = self.features(mode=m).iloc[train_indx, :], clss.iloc[test_indx]
            data_split.append((train_x, test_x, train_y, test_y))
        return data_split

    def cross_validation(self):
        for clss_set in self.cross_val_data:
            for splitted_data in clss_set:
                train_x, test_x, train_y, test_y = splitted_data
                self.__mlp.fit(X=train_x, y=train_y)
                self.__rfc.fit(X=train_x, y=train_y)
                self.__svc.fit(X=train_x, y=train_y)
                self.__bayes.fit(X=train_x, y=train_y)

        return NotImplemented

KFold(20, 0.7, 42, 1).cross_validation()
