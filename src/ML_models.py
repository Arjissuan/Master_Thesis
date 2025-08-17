from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class Models:
    def __init__(self):
        """
        Parameters:
        Multi Layered Perceptron: (64, 16, 8, 4, 2)_lbfgs_relu
        Random Forest Classiefier: 200_10_gini_log2
        Support Vector Machine: 1_sigmoid_scale_0.9_200_5
        """
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 16, 8, 4, 2), solver='lbfgs', activation='relu', max_iter=100000, early_stopping=True)
        self.rfc = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=10, max_features='log2')
        self.svc = SVC(C=1, kernel='sigmoid', gamma='scale', coef0=0.9, cache_size=200, degree=5)
        self.bayes = GaussianNB()