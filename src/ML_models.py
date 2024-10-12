from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class Models:
    def __init__(self):
        self.__mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), solver='adam', activation='tanh') #optimal_parameters
        self.__rfc = RandomForestClassifier()
        self.__svc = SVC()
        self.__bayes = GaussianNB()