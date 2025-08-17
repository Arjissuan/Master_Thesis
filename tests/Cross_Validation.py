import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import os
from src.MachineLearningData import ML_data
from src.ML_models import Models
from sklearn.model_selection import StratifiedShuffleSplit


class KFold(ML_data):
    def __init__(self, n=10, t=0.7, r=42, m=2):
        super().__init__()
        self.models = Models()
        self.drop_na()
        self.cross_val_data = list(map(lambda x:self.data_split(n, t, r, m, x), self.labels()))


    def data_split(self, n:int, t:float, r:int, m:int, clss:pd.Series):
        """

        :param n: number of splits
        :param t: train size
        :param r: random state
        :param m: features mode look into, mode are sets of features, there are 4 sets numbered 1 to 4
        :param clss: which class used for classification
        :return: splitted data ready for cross-validaiton
        """
        sss = StratifiedShuffleSplit(n_splits=n, train_size=t, random_state=r)
        data_split = []
        for train_indx, test_indx in sss.split(self.features(mode=m), clss):
            test_x, test_y = self.features(mode=m).iloc[test_indx, :], clss.iloc[test_indx]
            train_x, train_y = self.features(mode=m).iloc[train_indx, :], clss.iloc[train_indx]
            data_split.append((train_x, test_x, train_y, test_y))
        return data_split

    def cross_validation(self):
        if not os.path.exists("./cross_val/"):
            os.mkdir("./cross_val/")

        classes_names = ('gram+', 'gram-', 'fungi', 'cancer', 'virus')

        for j, clss_set in enumerate(self.cross_val_data):
            data_pred = ("f1","recall", "accuracy", 'precision')
            cols = ['MLP', 'RandForest', 'SuppVectMach', 'Bayes']
            temp = np.zeros(shape=[4, 4, 10])
            for i, splitted_data in enumerate(clss_set):
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
            df_std.to_csv(f"./cross_val/{classes_names[j]}_std.csv")
            df_pred.to_csv(f"./cross_val/{classes_names[j]}_pred_metrics.csv")
        message= "Cross-Validation complete"
# average + standard deviation
        return message

KFold(10, 0.7, 42, 2).cross_validation()
