import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.MachineLearningData import ML_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


mld = ML_data().drop_na()
X_train, X_test, y_train, y_test = train_test_split(mld.features(mode=4), mld.classes()[1], test_size=0.3, random_state=14)

if __name__ == "__main__":
    activations = ["identity", "logistic", "tanh"]
    solvers = ["lbfgs", "sgd", 'adam']
    hidden_layer_sizes = [(15, 5, 3), (64, 16, 8, 4, 2), (64, 32, 16, 8, 4, 2), (256, 64, 16, 4)]
    predictions = {}
    for activation in activations:
        for solver in solvers:
            for i, hls in enumerate(hidden_layer_sizes):
                mlp = MLPClassifier(hidden_layer_sizes=hls, activation=activation, solver=solver, max_iter=500, random_state=42)
                mlp.fit(X_train, y_train)
                predictions[f"{activation}_{solver}_{i}"] = (mlp.predict(X_test))

    df = pd.DataFrame()
    for key, pred in predictions.items():
        stats = pd.Series({"type": key,
                           "acc": accuracy_score(y_test, pred),
                           "f1": f1_score(y_test, pred),
                           "recall": recall_score(y_test, pred),
                           "precision": precision_score(y_test, pred)
                           }
                          )
        df = pd.concat([df, stats], axis=1)

    df.to_csv("./mlp_test/test_values_grid.csv")