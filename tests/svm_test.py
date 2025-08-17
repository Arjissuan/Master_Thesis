import pandas as pd
from sklearn.svm import SVC
from src import MachineLearningData
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

if __name__ == "__main__":
    MLD = MachineLearningData.ML_data()
    MLD.drop_na()
    X_train, X_test, y_train, y_test = train_test_split(MLD.features(mode=3), MLD.labels()[1], test_size=0.3,
                                                        random_state=14)
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    gamma = ('auto', 'scale')
    C = (1, 2, 5, 10)
    degree = (3, 5, 10)
    cache_size = (200, 400, 1000)
    max_iter = (1,2,5)
    coeficient = (0.1, 0.4, 0.9)

    data_pred = {"f1": [], "recall": [], "accuracy": [], 'precision': []}
    col_names = []
    for kern in kernels:
        for gam in gamma:
            for c in C:
                for deg in degree:
                    for cache in cache_size:
                        for iter in max_iter:
                            for coef in coeficient:
                                if kern == 'poly':
                                    svm = SVC(C=c, kernel=kern, degree=deg, gamma=gam,
                                              coef0=coef, cache_size = cache, max_iter = iter)
                                    mess = f"{c}_{kern}_{deg}_{gam}_{coef}_{cache}_{iter}"
                                elif kern=='sigmoid':
                                    svm = SVC(C=c, kernel=kern, gamma=gam,
                                              coef0=coef, cache_size=cache, max_iter=iter)
                                    mess = f"{c}_{kern}_{gam}_{coef}_{cache}_{iter}"
                                elif kern=='rbf':
                                    svm = SVC(C=c, kernel=kern, gamma=gam, cache_size=cache, max_iter=iter)
                                    mess = f"{c}_{kern}_{gam}_{cache}_{iter}"
                                else:
                                    svm = SVC(C=c, kernel=kern, cache_size=cache, max_iter=iter)
                                    mess = f"{c}_{kern}_{cache}_{iter}"

                                svm.fit(X=X_train, y=y_train)
                                pred = svm.predict(X=X_test)
                                col_names.append(mess)
                                data_pred['f1'].append(f1_score(y_true=y_test, y_pred=pred))
                                data_pred['recall'].append(recall_score(y_true=y_test, y_pred=pred))
                                data_pred['accuracy'].append(accuracy_score(y_true=y_test, y_pred=pred))
                                data_pred['precision'].append(precision_score(y_true=y_test, y_pred=pred))
                                print(len(col_names))
    df = pd.DataFrame(data=data_pred, index=col_names)

    if not os.path.exists("./svm_test/"):
        os.mkdir("./svm_test/")
    df.to_csv(path_or_buf="./svm_test/svm.csv", sep=',')