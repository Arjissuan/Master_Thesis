from src import LData
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def percentage(array):
    return array.sum()*100/len(array)

if __name__ == "__main__":
    ld = LData()
    y = pd.DataFrame(ld.labels())
    perc = y.apply(func=percentage, axis='rows')
    # sns.color_palette('bright')
    # plt.pie(perc.values, labels=y.columns, autopct='%.0f%%')
    # plt.show()
    plt.bar(perc.index, perc.values)
    plt.ylabel("Percentage")
    plt.xlabel("Label Name")
    plt.grid()
    plt.show()

    print(perc)



