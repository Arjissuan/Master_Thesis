from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import os

def deep_learning_evaluate_model(model, dl, device, class_names=None, threshold=0.5, output_dir="results"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb.float())                    # raw outputs
            probs = torch.sigmoid(logits)                 # convert to probabilities
            preds = (probs > threshold).int()             # threshold at 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    # Stack batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # === Classification Report ===
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    # === Create output dir if missing ===
    os.makedirs(output_dir, exist_ok=True)

    # === Save report ===
    csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(csv_path, index=True)
    print(f"\n Classification report saved to: {csv_path}")

    # === Print summary to console ===
    print("\nClassification Report:")
    print(report_df.round(3))

    # === Multilabel Confusion Matrices ===
    cms = multilabel_confusion_matrix(all_labels, all_preds)
    print(cms)
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        cm = cms[i]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(class_names[i])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.show()


    return all_preds, all_labels, report_df
