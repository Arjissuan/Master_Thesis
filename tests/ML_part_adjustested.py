from src import ML_data, KFold
#ML with crossvalidation
if __name__ == "__main__":
    X = list(map(lambda x:ML_data().features(mode=x), range(5)))
    y = ML_data().labels()

