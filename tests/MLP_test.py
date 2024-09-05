from sklearn.neural_network import MLPClassifier
from src.MachineLearningData import ML_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os

mld = ML_data().drop_na()
X_train, X_test, y_train, y_test = train_test_split(mld.features(mode=4), mld.classes()[1], test_size=0.3, random_state=14)

if __name__ == "__main__":
    mlp1 = MLPClassifier(random_state=14)
    mlp2 = MLPClassifier(hidden_layer_sizes=(15, 5, 3), random_state=14)
    mlp3 = MLPClassifier(hidden_layer_sizes=(64, 16, 8, 4, 2), random_state=14)
    mlp4 = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), random_state=14)
    mlp5 = MLPClassifier(hidden_layer_sizes=(256, 64, 16, 4), random_state=14)
    mlp6 = MLPClassifier(activation="identity", random_state=14)
    mlp7 = MLPClassifier(activation="logistic", random_state=14)
    mlp8 = MLPClassifier(activation="tanh", random_state=14)
    mlp9 = MLPClassifier(solver="lbfgs", random_state=14)
    mlp10 = MLPClassifier(solver="sgd", random_state=14)

    predictions = [
        mlp1.fit(X_train, y_train),
        mlp2.fit(X_train, y_train),
        mlp3.fit(X_train, y_train),
        mlp4.fit(X_train, y_train),
        mlp5.fit(X_train, y_train),
        mlp6.fit(X_train, y_train),
        mlp7.fit(X_train, y_train),
        mlp8.fit(X_train, y_train),
        mlp9.fit(X_train, y_train),
        mlp10.fit(X_train, y_train)
    ]


    os.mkdir("./mlp_test")

    for i, pred in enumerate(predictions):
        stats = {"type":"SVC",
                         "acc":accuracy_score( y_test, pred),
                         "f1":f1_score(y_test, pred),
                         "recall":recall_score(y_test, pred),
                         "precision":precision_score(y_test, pred)
                         }