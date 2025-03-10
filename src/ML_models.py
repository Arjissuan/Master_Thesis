from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class Models:
    def __init__(self):
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), solver='adam', activation='tanh', max_iter=1000) #tanh_adam_(64, 32, 16, 8, 4, 2)
        self.rfc = RandomForestClassifier(n_estimators=400, criterion="log_loss", max_depth=40, max_features='log2') #400_40_log_loss_log2
        self.svc = SVC()
        self.bayes = GaussianNB()