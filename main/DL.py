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

    labels = pd.DataFrame(list(map(lambda x: [
        int(classes["antigram+"][x]),
        int(classes["antigram-"][x]),
        int(classes["antifungal"][x]),
        int(classes["anticancer"][x]),
        int(classes["antivirial"][x])
    ], classes["antigram+"].index)))

    labels = labels.astype(int)
    maximal_lenght = np.max(features.loc[:, 'Length'])


    #
    # # -------------------------------------------------
    # # 2. DataLoaders
    # # -------------------------------------------------
    splitter = DLDataSplit(X=features, Y=labels, batch_size=32, max_len=maximal_lenght)
    train_dl, val_dl, test_dl = splitter.get_loaders()
    # print(train_dl)
    #
    #
    # # -------------------------------------------------
    # # 3. Model
    # # -------------------------------------------------

    num_classes = labels.shape[1]
    print("Final num_classes:", num_classes, type(num_classes))
    # should be 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeptideCNN(seq_len=maximal_lenght, num_classes=num_classes, debug=False)


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
