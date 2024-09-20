import pandas as pd
from sklearn.neural_network import MLPClassifier
from src.MachineLearningData import ML_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


mld = ML_data().drop_na()
X_train, X_test, y_train, y_test = train_test_split(mld.features(mode=4), mld.classes()[1], test_size=0.3, random_state=14)

if __name__ == "__main__":
    # rnd_state = 41
    # mlp1 = MLPClassifier(random_state=rnd_state, max_iter=500)
    # mlp2 = MLPClassifier(hidden_layer_sizes=(15, 5, 3), random_state=rnd_state, max_iter=500)
    # mlp3 = MLPClassifier(hidden_layer_sizes=(64, 16, 8, 4, 2), random_state=rnd_state, max_iter=500)
    # mlp4 = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8, 4, 2), random_state=rnd_state, max_iter=500)
    # mlp5 = MLPClassifier(hidden_layer_sizes=(256, 64, 16, 4), random_state=rnd_state, max_iter=500)
    # mlp6 = MLPClassifier(activation="identity", random_state=rnd_state, max_iter=500)
    # mlp7 = MLPClassifier(activation="logistic", random_state=rnd_state, max_iter=500)
    # mlp8 = MLPClassifier(activation="tanh", random_state=rnd_state, max_iter=500)
    # mlp9 = MLPClassifier(solver="lbfgs", random_state=rnd_state, max_iter=500)
    # mlp10 = MLPClassifier(solver="sgd", random_state=rnd_state, max_iter=500)
    #
    # mlp1.fit(X_train, y_train)
    # mlp2.fit(X_train, y_train)
    # mlp3.fit(X_train, y_train)
    # mlp4.fit(X_train, y_train)
    # mlp5.fit(X_train, y_train)
    # mlp6.fit(X_train, y_train)
    # mlp7.fit(X_train, y_train)
    # mlp8.fit(X_train, y_train)
    # mlp9.fit(X_train, y_train)
    # mlp10.fit(X_train, y_train)
    #
    # predictions = [
    #     mlp1.predict(X_test),
    #     mlp2.predict(X_test),
    #     mlp3.predict(X_test),
    #     mlp4.predict(X_test),
    #     mlp5.predict(X_test),
    #     mlp6.predict(X_test),
    #     mlp7.predict(X_test),
    #     mlp8.predict(X_test),
    #     mlp9.predict(X_test),
    #     mlp10.predict(X_test),
    # ]
    #
    # df = pd.DataFrame()
    # for i, pred in enumerate(predictions):
    #     stats = pd.Series({"type":i+1,
    #                      "acc":accuracy_score( y_test, pred),
    #                      "f1":f1_score(y_test, pred),
    #                      "recall":recall_score(y_test, pred),
    #                      "precision":precision_score(y_test, pred)
    #                      })
    #     df = pd.concat([df, stats], axis=1)
    #
    # df.to_csv("./mlp_test/test_values.csv")

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