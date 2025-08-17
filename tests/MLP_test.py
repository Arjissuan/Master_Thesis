import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.MachineLearningData import ML_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import subprocess

mld = ML_data().drop_na()
X_train, X_test, y_train, y_test = train_test_split(mld.features(mode=4), mld.labels()[1], test_size=0.3, random_state=14)

if __name__ == "__main__":
    activations = ["identity", "logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", 'adam']
    hidden_layer_sizes = [(15, 5, 3), (64, 16, 8, 4, 2), (64, 32, 16, 8, 4, 2), (256, 64, 16, 4)]
    df = pd.DataFrame()
    for act in activations:
        for sol in solvers:
            for hls in hidden_layer_sizes:
                MLP = MLPClassifier(hidden_layer_sizes=hls, solver=sol, activation=act, max_iter=200000, early_stopping=True, validation_fraction=0.2, warm_start=True)
                classifier_name = f"{hls}_{sol}_{act}"
                MLP.fit(X=X_train, y=y_train)
                pred = MLP.predict(X_test)
                row = pd.DataFrame(data=[accuracy_score(y_true=y_test, y_pred=pred),
                                         f1_score(y_true=y_test, y_pred=pred),
                                         recall_score(y_true=y_test, y_pred=pred),
                                         precision_score(y_true=y_test, y_pred=pred)],
                                   index=["accuracy", "f1", "recall", "precision"],
                                   columns=[classifier_name,])
                df = pd.concat([df, row], axis=1)

#do poprawy

    df.T.to_csv("./mlp_test/mlp.csv")
    subprocess.run(["shutdown"])