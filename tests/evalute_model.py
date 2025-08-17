import pandas as pd
import numpy as np
import os

def evaluate(gsdf:pd.DataFrame, scoring_weights=None):
    """

    :param gsdf: Grid search dataframe
    :return: best model string, best model metrics
    """
    assert type(gsdf)==pd.DataFrame, "This is not dataframe"
    if scoring_weights is None:
        scoring_weights = {
            'accuracy':0,
            'f1':1,
            'recall':0,
            'precision':0
        }

    total = np.sum(scoring_weights.values())
    # scoring_weights = {k: v / total for k,v in scoring_weights.items()}

    gsdf['score'] = (scoring_weights['accuracy'] * gsdf['accuracy'] +
        scoring_weights['f1'] * gsdf['f1'] +
        scoring_weights['recall'] * gsdf['recall'] +
        scoring_weights['precision'] * gsdf['precision']
    )
    best_idx = gsdf['score'].idxmax()
    best_row = gsdf.loc[best_idx]
    return best_idx, best_row

for path in ("mlp_test", "random_forest_test", "svm_test"):

    df = pd.read_csv(os.path.join(f'./{path}', os.listdir(f'./{path}')[0]), index_col=0)

    print(evaluate(df)[0])