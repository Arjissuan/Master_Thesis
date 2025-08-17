from src import KFold, ML_data
import os


if __name__ == "__main__":
    MLD = ML_data()
    MLD.drop_na()
    X = list(map(lambda x:MLD.features(mode=x), range(1,5)))
    y = MLD.labels()
    if not os.path.exists("cross_validation_results/"):
        os.mkdir("cross_validation_results")
    # for index, i in enumerate(X):
    #     print()
    #     for j in y.keys():
    #         pred_df, std_df = KFold(labels=y[j], features=i).cross_validation()
    #         pred_df.to_csv(f"cross_validation_results/pred_mode_{index}_{j}.csv")
    #         std_df.to_csv(f"cross_validation_results/std_mode_{index}_{j}.csv")



    for index in range(1,5):
        df = MLD.cancer_df(mode=index)
        X_cancer, y_cancer = df['features'], df['labels']
        canc_pred_df, canc_std_df = KFold(labels=y_cancer, features=X_cancer).cross_validation()
        canc_pred_df.to_csv(f"cross_validation_results/cancer_{index}_pred.csv")
        canc_std_df.to_csv(f"cross_validation_results/cancer{index}_std.csv")