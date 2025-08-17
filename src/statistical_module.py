import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.patheffects as path_effects
import seaborn as sn
from src.MachineLearningData import ML_data
from sklearn.feature_selection import RFE

class StatisticalModule:

    def __init__(self, data: pd.DataFrame, label: pd.Series, alfa=0.05) -> None:
        """

        :param data: Dataframe of relevant features
        :param label: Series of label that you want to test
        :param alfa: alfa, set to 0.05
        """
        self.data = data #
        self.label = label # explanatory variable
        self.categorical = self.data.loc[:, self.data.dtypes==object].columns
        self.numerical = self.data.loc[:, (self.data.dtypes!=object) & np.logical_not(self.data.isin([0,1]).all())].columns
        self.binary = self.data.loc[:, (self.data.dtypes!=object) & (self.data.isin([0,1]).all())].columns
        self.alfa = alfa
        self.m = len(self.numerical)

    def RecursiveFeatureElimination(self):
        return None

    def UnivariateFeatureSelection(self):
        return NotImplemented


    def DescriptiveStatistics(self):
        statistic_df = pd.DataFrame([self.data.skew(axis='columns'),
                                  self.data.mean(axis='columns'),
                                  self.data.mode(axis='columns'),
                                  self.data.median(axis='columns'),
                                  self.data.var(axis='columns'),
                                  self.data.std(axis='columns'),
                                  ], index=["Skewness",
                                            'Mode',
                                            'Median',
                                            'Variance',
                                            'Standard Deviation'])
        return NotImplemented

    #Normalcy
    def ShapiroWilk(self):
        """
        h0: Data is from normal distribution
        ha: Data is not from normal distribution
        """
        results = stats.shapiro(self.data.loc[:, self.numerical], axis=0)
        nomralcy_table = pd.DataFrame(data=results, columns=self.numerical, index=['statistic', 'pvalue']).T
        Intepret = list(map(lambda x: ("h0 embraced" if x > self.alfa else "rejected h0"), nomralcy_table['pvalue']))
        nomralcy_table["Intepret"] = Intepret
        return nomralcy_table

    def KolmogorowSmirnof(self):
        """
        h0: Data is from normal distribution
        ha: Data is not from normal distribution
        """
        result = stats.kstest(self.data.loc[:, self.numerical], 'norm', axis=0)
        normalcy_table = pd.DataFrame(data=result, columns=self.numerical, index=['statistic', 'pvalue']).T
        Intepret = list(map(lambda x: ("h0 embraced" if x > self.alfa else "rejected h0"), normalcy_table['pvalue']))
        normalcy_table["Intepret"] = Intepret
        return normalcy_table

    #checking variance equality
    def lavene(self):
        """
        Use when data is not from normal distribution
        h0: All imput samples are from populations with equal variances
        ha: Not all imput samples are from populations with equal variances
        :return:
        """
        result = {}
        for i in self.numerical:
            for j in self.numerical:
                if i != j:
                    result[f"{i}_{j}"] = stats.levene(self.data.loc[:, i], self.data.loc[:, j])
        lav_df = pd.DataFrame(result, index=["Satistic", "pvalue"]).T
        lav_df["Interpretation"] = list(map(lambda x:("Failed to reject h0" if x>self.alfa else "rejceted h0"), lav_df["pvalue"]))
        return lav_df

    #Multiple testing correction
    def bonferroni(self, p):
        return min(p*self.m, 1.0)

    def sidek(self,p):
        return 1-(1-p)**self.m

    def MultipleTestingCorrection(self):
        """
        h0: The variable is not associated with explanatory variable
        :return: The variable is associated with explanatory variable
        """
        p_values = []
        for variable in self.numerical:
            contigency_table = pd.crosstab(self.data[variable] > self.data[variable].mean(), self.label)
            _, p, __, ___ = chi2_contingency(contigency_table)
            p_values.append(p)
        assert len(p_values) == len(self.numerical), "not equal"
        bonferroni_corected = list(map(self.bonferroni, p_values))
        sidak_corrected = list(map(self.sidek, p_values))

        results = pd.DataFrame({
            'Variable': self.numerical,
            'Uncorrected p': p_values,
            "Bonferroni corrected": bonferroni_corected,
            "Sidak corrected": sidak_corrected,
        })
        results = results.sort_values("Uncorrected p").reset_index(drop=True)
        results['Rank'] = np.arange(1, self.m + 1)
        results["Benjamini-Hochberg corrected"] = results['Uncorrected p'] * self.m / results['Rank']
        results["Benjamini-Hochberg corrected"] = np.minimum.accumulate(results["Benjamini-Hochberg corrected"][::-1])[
                                                  ::-1]
        results = results.sort_values("Rank").drop(columns=['Rank']).reset_index(drop=True)
        return results

    def EffectSize(self):
        effect_sizes = []
        # Calculate Wendt's A-value for each variable
        for variable in self.numerical:
            group_yes = self.data[self.label== 1][variable]
            group_no = self.data[self.label == 0][variable]

            # Perform Mann-Whitney U test
            u_statistic, _ = mannwhitneyu(group_yes, group_no, alternative="two-sided")
            n1 = len(group_yes)
            n2 = len(group_no)
            wendt_a = 1 - ((2 * min(u_statistic, (n1 * n2 - u_statistic))) / (n1 * n2)) if n1 > 0 and n2 > 0 else None

            effect_sizes.append({"Variable": variable, "Effect_Size": wendt_a})

        results_df = pd.DataFrame(effect_sizes)

        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Variable"], results_df["Effect_Size"], color="skyblue")
        plt.axhline(y=0.1, color="green", linestyle="--", linewidth=3)
        plt.axhline(y=0.3, color="#adac3c", linestyle="--", linewidth=3)
        plt.axhline(y=0.5, color="orange", linestyle="--", linewidth=3)
        plt.axhline(y=0.7, color="red", linestyle="--", linewidth=3)

        # Add text with black outline
        def add_outlined_text(x, y, text, color, fontsize):
            t = plt.text(x, y, text, color=color, fontsize=fontsize, ha='right', path_effects=[
                path_effects.Stroke(linewidth=0.4, foreground="black"),
                path_effects.Normal()
            ])
            return t

        add_outlined_text(len(results_df["Variable"]) - 0.5, 0.1 + 0.01, "Small effect size", "green", 15)
        add_outlined_text(len(results_df["Variable"]) - 0.5, 0.3 + 0.01, "Medium effect size", "#adac3c", 15)
        add_outlined_text(len(results_df["Variable"]) - 0.5, 0.5 + 0.01, "Large effect size", "orange", 15)
        add_outlined_text(len(results_df["Variable"]) - 0.5, 0.7 + 0.01, "Very large effect size", "red", 15)

        plt.title("Wendt's A Effect Size for Variables Grouped by 'RainTomorrow'", fontsize=16)
        plt.ylabel("Effect Size (A-value)", fontsize=16)
        plt.xlabel("Variables", fontsize=16)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        return results_df

    def Corelations(self):
        corr_matrix = self.data.loc[:, self.numerical].corr()
        sn.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
        plt.show()


if __name__ == "__main__":
    mld = ML_data()
    SM = StatisticalModule(mld.features(4), mld.labels()[0], )
    print(SM.numerical, SM.label)
    SM.EffectSize()
    SM.Corelations()
