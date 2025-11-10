import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, roc_auc_score
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
        self.labels = labels
        self.features = features


    def data_split(self, n:int, t:float, r:int, lbls:pd.Series, feat: pd.DataFrame)->List:
        """
        Splits data for cross validation
        :param n: number of splits
        :param t: train size
        :param r: random state
        :param lbls: labels used for classification
        :param feat: featurs used for classification
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
           Performs cross-validation across all models.

           :return:
               df_pred: DataFrame with mean performance metrics
               df_std: DataFrame with std performance metrics
               all_true: dict of all true labels per model
               all_pred: dict of all predicted labels per model
        """
        if not os.path.exists("./cross_val/"):
            os.mkdir("./cross_val/")

        # classes_names = labels.keys()

        metrics = ("f1", "recall", "accuracy", "precision", "roc_auc")
        models_names = ['MLP', 'RandForest', 'SuppVectMach', 'Bayes']
        num_metrics, num_models, num_folds = len(metrics), len(models_names), len(self.cross_val_data)
        temp = np.zeros(shape=[num_metrics, num_models, num_folds])

        all_true = {m: [] for m in models_names}
        all_pred = {m: [] for m in models_names}

        for i, (train_x, test_x, train_y, test_y) in enumerate(self.cross_val_data):
            # Fit
            self.models.mlp.fit(X=train_x, y=train_y)
            self.models.rfc.fit(X=train_x, y=train_y)
            self.models.svc.fit(X=train_x, y=train_y)
            self.models.bayes.fit(X=train_x, y=train_y)

            # Predict
            mlp_pred = self.models.mlp.predict(X=test_x)
            rfc_pred = self.models.rfc.predict(X=test_x)
            svc_pred = self.models.svc.predict(X=test_x)
            bayes_pred = self.models.bayes.predict(X=test_x)

            preds = [mlp_pred, rfc_pred, svc_pred, bayes_pred]

            # Store all true/predicted
            for name, pred in zip(models_names, preds):
                all_true[name].extend(test_y)
                all_pred[name].extend(pred)

            # --- Compute metrics ---
            for m_idx, pred in enumerate(preds):
                temp[0, m_idx, i] = f1_score(test_y, pred)
                temp[1, m_idx, i] = recall_score(test_y, pred)
                temp[2, m_idx, i] = accuracy_score(test_y, pred)
                temp[3, m_idx, i] = precision_score(test_y, pred)

                try:
                    if hasattr(self.models, models_names[m_idx].lower()) and hasattr(
                            getattr(self.models, models_names[m_idx].lower()), "predict_proba"
                    ):
                        proba = getattr(self.models, models_names[m_idx].lower()).predict_proba(test_x)[:, 1]
                        temp[4, m_idx, i] = roc_auc_score(test_y, proba)
                    else:
                        temp[4, m_idx, i] = np.nan
                except Exception:
                    temp[4, m_idx, i] = np.nan

            # --- Aggregate results ---
        std_matrix = np.std(temp, axis=2)
        mean_matrix = np.mean(temp, axis=2)
        df_pred = pd.DataFrame(data=mean_matrix, index=metrics, columns=models_names)
        df_std = pd.DataFrame(data=std_matrix, index=metrics, columns=models_names)

        print("Cross-Validation complete")

        return df_pred, df_std, all_true, all_pred