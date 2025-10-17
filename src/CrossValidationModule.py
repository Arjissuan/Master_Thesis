import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import os
from src.ML_models import Models
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List


class KFold:
    def __init__(self, labels:pd.Series, features:pd.DataFrame, n=10, t=0.9, r=42):
        """
        Initilization gives ready to use data for cross validation,
        which can be performed using already implemented method.

        :param n: number of splits
        :param t: train size
        :param r: random state
        """
        super().__init__()
        self.models = Models()
        self.cross_val_data = self.data_split(n, t, r, labels, features)


    def data_split(self, n:int, t:float, r:int, lbls:pd.Series, feat: pd.DataFrame)->List:
        """
        Splits data for cross validation
        :param n: number of splits
        :param t: train size
        :param r: random state
        :param lbls: which class used for classification
        :return: splitted data ready for cross-validaiton
        """
        sss = StratifiedShuffleSplit(n_splits=n, train_size=t, random_state=r)
        data_split = []
        for train_indx, test_indx in sss.split(feat, lbls):
            test_x, test_y = feat.iloc[test_indx, :], lbls.iloc[test_indx]
            train_x, train_y = feat.iloc[train_indx, :], lbls.iloc[train_indx]
            data_split.append((train_x, test_x, train_y, test_y))
        return data_split

    def cross_validation(self):
        """
        Performs cross validation
        :return: Message informing that process ended.
        """
        if not os.path.exists("./cross_val/"):
            os.mkdir("./cross_val/")

        # classes_names = labels.keys()


        data_pred = ("f1","recall", "accuracy", 'precision')
        cols = ['MLP', 'RandForest', 'SuppVectMach', 'Bayes']
        temp = np.zeros(shape=[4, 4, 10])
        for i, splitted_data in enumerate(self.cross_val_data):
            train_x, test_x, train_y, test_y = splitted_data

            self.models.mlp.fit(X=train_x, y=train_y)
            self.models.rfc.fit(X=train_x, y=train_y)
            self.models.svc.fit(X=train_x, y=train_y)
            self.models.bayes.fit(X=train_x, y=train_y)

            mlp_pred = self.models.mlp.predict(X=test_x)
            rfc_pred = self.models.rfc.predict(X=test_x)
            svc_pred = self.models.svc.predict(X=test_x)
            bayes_pred = self.models.bayes.predict(X=test_x)

            temp[0,0,i] = f1_score(y_true=test_y, y_pred=mlp_pred)
            temp[0,1,i] = f1_score(y_true=test_y, y_pred=rfc_pred)
            temp[0,2,i] = f1_score(y_true=test_y, y_pred=svc_pred)
            temp[0,3,i] = f1_score(y_true=test_y, y_pred=bayes_pred)

            temp[1,0,i] = recall_score(y_true=test_y, y_pred=mlp_pred)
            temp[1,1,i] = recall_score(y_true=test_y, y_pred=rfc_pred)
            temp[1,2,i] = recall_score(y_true=test_y, y_pred=svc_pred)
            temp[1,3,i] = recall_score(y_true=test_y, y_pred=bayes_pred)

            temp[2,0,i] = accuracy_score(y_true=test_y, y_pred=mlp_pred)
            temp[2,1,i] = accuracy_score(y_true=test_y, y_pred=rfc_pred)
            temp[2,2,i] = accuracy_score(y_true=test_y, y_pred=svc_pred)
            temp[2,3,i] = accuracy_score(y_true=test_y, y_pred=bayes_pred)

            temp[3,0,i] = precision_score(y_true=test_y, y_pred=mlp_pred)
            temp[3,1,i] = precision_score(y_true=test_y, y_pred=rfc_pred)
            temp[3,2,i] = precision_score(y_true=test_y, y_pred=svc_pred)
            temp[3,3,i] = precision_score(y_true=test_y, y_pred=bayes_pred)

        std_matrix = np.std(temp, axis=2)
        mean_matrix = np.mean(temp, axis=2)
        df_pred = pd.DataFrame(data=mean_matrix, index=data_pred, columns=cols)
        df_std = pd.DataFrame(data=std_matrix, index=data_pred, columns=cols)
        message= "Cross-Validation complete"
        print(message)
# average + standard deviation
        return df_pred, df_std
