from src.MachineLearningData import ML_data
from sklearn.model_selection import StratifiedShuffleSplit


mld = ML_data().drop_na()

def cross_validation(class_vect, splits, test_s, r_state):
    threeS = StratifiedShuffleSplit(n_splits=splits, test_size=test_s, random_state=r_state)
    splitted_data = []
    for indexes in threeS.split(mld.features(), class_vect):
        X_train, X_test, y_train, y_test = mld.features().iloc[indexes[0],:], mld.features().iloc[indexes[1],:], class_vect.iloc[indexes[0], :], class_vect.iloc[indexes[1],:]
        splitted_data.append([X_train, X_test, y_train, y_test])
    return splitted_data

def CV_classification(clss):
    return NotImplemented