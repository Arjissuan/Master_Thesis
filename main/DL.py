from src.DL_training_loop import run_model
from src import deep_learning_evaluate_model, DLDataSplit, PeptideLinear, PeptideCNN, LData
import torch
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # -------------------------------------------------
    # 1. Dataset
    # -------------------------------------------------

    ldata = LData()
    features = ldata.features_df[["Sequence", "Length"]]

    classes = ldata.labels()  # dictionary of series
    num_samples = len(ldata.features_df)
    multi_class_labels = pd.Series(np.zeros(num_samples, dtype=int), index=ldata.features_df.index)

    priority = ["antigram+", "antigram-", "antifungal", "anticancer", "antivirial"]

    for idx in ldata.features_df.index:
        for i, cls in enumerate(priority):
            if classes[cls].loc[idx]:
                multi_class_labels.loc[idx] = i
                break

    labels = multi_class_labels
    print(labels.head())
    print(features.head())

    # -------------------------------------------------
    # 2. DataLoaders
    # -------------------------------------------------
    splitter = DLDataSplit(features, labels, batch_size=32, max_len=None)
    train_dl, val_dl, test_dl = splitter.__getitem__()

    # -------------------------------------------------
    # 3. Model
    # -------------------------------------------------
    seq_len = train_dl.dataset.seq_len   # dataset knows max_len
    num_classes = labels.nunique()       # should be 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try either baseline linear model or CNN
    # model = PeptideLinear(seq_len, num_classes=num_classes)
    model = PeptideCNN(seq_len, num_classes=num_classes)

    model = model.to(device)

    # -------------------------------------------------
    # 4. Train model
    # -------------------------------------------------
    history = run_model(
        train_dl, val_dl, model, device,
        lr=1e-3,
        epochs=20
    )

    # -------------------------------------------------
    # 5. Evaluate model
    # -------------------------------------------------
    class_names = ["Gram+", "Gram-", "Fungi", "Virus", "Cancer"]

    print("\nValidation set performance:")
    deep_learning_evaluate_model(model, val_dl, device, class_names)

    print("\nTest set performance:")
    deep_learning_evaluate_model(model, test_dl, device, class_names)