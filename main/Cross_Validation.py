from src import KFold, ML_data
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    accuracy_score
)


if __name__ == "__main__":
    MLD = ML_data()
    X = [MLD.features(mode=x) for x in range(1, 5)]
    y = MLD.labels()


    if not os.path.exists("cross_validation_results/"):
        os.mkdir("cross_validation_results")

    for mode_index, features in enumerate(X):
        print(f"\nProcessing feature mode {mode_index + 1}...")

        # Storage for global micro/macro computation
        all_true_by_model = {"MLP": [], "RandForest": [], "SuppVectMach": [], "Bayes": []}
        all_pred_by_model = {"MLP": [], "RandForest": [], "SuppVectMach": [], "Bayes": []}
        per_label_results = {}

        # ---- Run cross-validation for each label ----
        for label_name, label_values in y.items():
            print(f"  → Cross-validating label: {label_name}")

            pred_df, std_df, all_true, all_pred = KFold(labels=label_values, features=features).cross_validation()

            # Save per-label metrics
            pred_df.to_csv(f"cross_validation_results/pred_mode_{mode_index}_{label_name}.csv")
            std_df.to_csv(f"cross_validation_results/std_mode_{mode_index}_{label_name}.csv")

            # Store results for macro average
            per_label_results[label_name] = pred_df

            # Extend true/pred labels for micro average
            for model_name in all_true_by_model.keys():
                all_true_by_model[model_name].extend(all_true[model_name])
                all_pred_by_model[model_name].extend(all_pred[model_name])

        # ---- MACRO AVERAGE across labels ----
        # Average per-label DataFrames (equal weight per label)
        macro_avg_df = pd.concat(per_label_results.values()).groupby(level=0).mean()
        macro_avg_path = f"cross_validation_results/macro_avg_mode_{mode_index}.csv"
        macro_avg_df.to_csv(macro_avg_path)
        print(f"  Saved macro-average metrics to {macro_avg_path}")

        # ---- MICRO AVERAGE across labels ----
        micro_records = []
        for model_name in all_true_by_model.keys():
            y_true_all = np.array(all_true_by_model[model_name])
            y_pred_all = np.array(all_pred_by_model[model_name])

            if len(np.unique(y_true_all)) < 2:
                print(f"  Skipping micro average for {model_name} (only one class present).")
                continue

            micro_metrics = {
                "model": model_name,
                "f1_micro": f1_score(y_true_all, y_pred_all, average="micro"),
                "precision_micro": precision_score(y_true_all, y_pred_all, average="micro"),
                "recall_micro": recall_score(y_true_all, y_pred_all, average="micro"),
                "accuracy": accuracy_score(y_true_all, y_pred_all),
            }
            micro_records.append(micro_metrics)

        micro_avg_df = pd.DataFrame(micro_records)
        micro_avg_path = f"cross_validation_results/micro_avg_mode_{mode_index}.csv"
        micro_avg_df.to_csv(micro_avg_path, index=False)
        print(f"  Saved micro-average metrics to {micro_avg_path}")


# if __name__ == "__main__":
#     MLD = ML_data()
#     # MLD.drop_na()
#     X = list(map(lambda x:MLD.features(mode=x), range(1,5)))
#     y = MLD.labels()
#     if not os.path.exists("cross_validation_results/"):
#         os.mkdir("cross_validation_results")
#     for index, i in enumerate(X):
#         for j in y.keys():
#             pred_df, std_df = KFold(labels=y[j], features=i).cross_validation()
#             pred_df.to_csv(f"cross_validation_results/pred_mode_{index}_{j}.csv")
#             std_df.to_csv(f"cross_validation_results/std_mode_{index}_{j}.csv")
#
#
#
#     for index in range(1,5):
#         df = MLD.cancer_df(mode=index)
#         X_cancer, y_cancer = df['features'], df['labels']
#         canc_pred_df, canc_std_df = KFold(labels=y_cancer, features=X_cancer).cross_validation()
#         canc_pred_df.to_csv(f"cross_validation_results/cancer_{index}_pred.csv")
#         canc_std_df.to_csv(f"cross_validation_results/cancer{index}_std.csv")