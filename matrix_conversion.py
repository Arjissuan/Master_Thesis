import pandas as pd


def matrix_conversion():
    df = pd.read_excel("AMP_30_03_2020 IMPROVED.xlsx")

    new_df = pd.DataFrame(columns=["ID", "Sequence", "Alanine", "Arginine", "Asparagine", "Aspartic_Acid", "Cysteine",
                                   "Glutamic_Acid", "Glutamine", "Glycine", "Histidine", "Isoleucine", "Leucine",
                                   "Lysine", "Mathionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan",
                                   "Tyrosine", "Valine"])

    code_letters = ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

    new_df["Sequence"] = df["Sequence"].copy()
    new_df["ID"] = df["ID"].copy()

    for i, col in enumerate(new_df.columns[2:]):
        new_df[col] = list(map(lambda x: x.count(code_letters[i]), new_df["Sequence"]))
    print(new_df)
    return new_df

df = matrix_conversion()
df.to_csv("ML_AMP.csv")