import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src import MachineLearningData
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
#it is too slow
if __name__ == "__main__":
    n_tree = list(map(lambda x: 100**x, range(3)))
    crit = ('gini', 'entropy', 'log_loss')
    max_depth = list(map(lambda x: 10**x, range(3)))
    min_sam_split = list(map(lambda x: 2+x, range(3)))
    min_sample_leaf = list(map(lambda x: 1+x, range(3)))
    max_feat = ("sqrt", "log2", None)
    MLD = MachineLearningData.ML_data()
    MLD.drop_na()
    X_train, X_test, y_train, y_test = train_test_split(MLD.features(mode=3), MLD.classes()[1], test_size=0.3,
                                                        random_state=14)
    data_pred = {"f1":[], "recall":[], "accuracy":[], 'precision':[]}
    col_names = []
    for tree in n_tree:
        for depth in max_depth:
            for split in min_sam_split:
                for leaf in min_sample_leaf:
                    for criteria in crit:
                        for feat in max_feat:
                            RFC = RandomForestClassifier(n_estimators=tree,
                                                         criterion=criteria,
                                                         max_depth=depth,
                                                         min_samples_split=split,
                                                         min_samples_leaf=leaf,
                                                         max_features=feat,
                                                         )
                            RFC.fit(X=X_train, y=y_train)
                            pred = RFC.predict(X=X_test)
                            col_names.append(f"{tree}_{depth}_{split}_{leaf}_{criteria}_{feat}")
                            data_pred['f1'].append(f1_score(y_true=y_test, y_pred=pred))
                            data_pred['recall'].append(recall_score(y_true=y_test, y_pred=pred))
                            data_pred['accuracy'].append(accuracy_score(y_true=y_test, y_pred=pred))
                            data_pred['precision'].append(precision_score(y_true=y_test, y_pred=pred))
                            print(len(col_names))
    df = pd.DataFrame(data=data_pred, index=col_names)

    if not os.path.exists("./random_forest_test/"):
        os.mkdir("./random_forest_test/")
    df.to_csv(path_or_buf="./random_forest_test/rft.csv", sep=',')
