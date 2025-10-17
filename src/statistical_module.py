import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu


class StatisticalModule:
    """
    Comprehensive statistical analysis module for mixed-type biological datasets.

    Features:
        - Automatic identification of categorical, binary, and numerical variables
        - Descriptive statistics per variable
        - Normality testing (Shapiro–Wilk)
        - Homogeneity of variances (Levene’s test)
        - Multiple testing correction (Bonferroni, Sidak, Benjamini–Hochberg)
        - Effect size estimation (Wendt’s A via Mann–Whitney U)
        - Spearman correlation for feature redundancy analysis
        - Optional summary report aggregation
    """

    def __init__(self, data: pd.DataFrame, label: pd.Series, alpha: float = 0.05):
        self.data = data.copy()
        self.label = label
        self.alpha = alpha

        # Variable type identification
        self.categorical = self.data.select_dtypes(include=["object"]).columns.tolist()
        self.binary = [
            col for col in self.data.select_dtypes(exclude=["object"]).columns
            if set(self.data[col].unique()) <= {0, 1}
        ]
        self.numerical = [
            col for col in self.data.columns
            if col not in self.categorical + self.binary
        ]

        self.m = len(self.numerical)  # number of numeric features

    # ---------------------------------------------------------------------- #
    #  Descriptive statistics
    # ---------------------------------------------------------------------- #
    def descriptive_statistics(self) -> pd.DataFrame:
        stats_df = pd.DataFrame({
            'Mean': self.data[self.numerical].mean(),
            'Median': self.data[self.numerical].median(),
            'Mode': self.data[self.numerical].mode().iloc[0],
            'Variance': self.data[self.numerical].var(),
            'StdDev': self.data[self.numerical].std(),
            'Skewness': self.data[self.numerical].skew(),
            'Kurtosis': self.data[self.numerical].kurt()
        })
        return stats_df.round(4)

    # ---------------------------------------------------------------------- #
    #  Normality test
    # ---------------------------------------------------------------------- #
    def shapiro_wilk(self) -> pd.DataFrame:
        results = []
        for col in self.numerical:
            stat, p = stats.normaltest(self.data[col])
            results.append({
                'Variable': col,
                'Statistic': stat,
                'p-value': p,
                'Normality': 'Normal' if p > self.alpha else 'Non-normal'
            })
        return pd.DataFrame(results)

    # ---------------------------------------------------------------------- #
    #  Homogeneity of variances (Levene)
    # ---------------------------------------------------------------------- #
    def levene_test(self) -> pd.DataFrame:
        results = []
        for col in self.numerical:
            group0 = self.data[self.label == 0][col]
            group1 = self.data[self.label == 1][col]
            if len(group0) > 1 and len(group1) > 1:
                stat, p = stats.levene(group0, group1)
                results.append({
                    'Variable': col,
                    'Statistic': stat,
                    'p-value': p,
                    'Equal Variance': 'Yes' if p > self.alpha else 'No'
                })
        return pd.DataFrame(results)

    # ---------------------------------------------------------------------- #
    #  Multiple testing correction
    # ---------------------------------------------------------------------- #
    def multiple_testing_correction(self) -> pd.DataFrame:
        p_values = []
        for var in self.numerical:
            cont = pd.crosstab(self.data[var] > self.data[var].mean(), self.label)
            _, p, _, _ = chi2_contingency(cont)
            p_values.append(p)

        results = pd.DataFrame({
            'Variable': self.numerical,
            'Uncorrected p': p_values
        })

        results['Bonferroni'] = np.minimum(results['Uncorrected p'] * self.m, 1.0)
        results['Sidak'] = 1 - (1 - results['Uncorrected p']) ** self.m

        # Benjamini–Hochberg FDR
        results = results.sort_values("Uncorrected p").reset_index(drop=True)
        results['Rank'] = np.arange(1, self.m + 1)
        results['Benjamini–Hochberg'] = (
            results['Uncorrected p'] * self.m / results['Rank']
        ).clip(upper=1.0)
        results['Benjamini–Hochberg'] = np.minimum.accumulate(
            results['Benjamini–Hochberg'][::-1]
        )[::-1]

        return results.sort_values("Variable").reset_index(drop=True)

    # ---------------------------------------------------------------------- #
    #  Effect size (Wendt’s A via Mann–Whitney U)
    # ---------------------------------------------------------------------- #
    def effect_size(self, show_plot=True) -> pd.DataFrame:
        results = []
        for var in self.numerical:
            g1 = self.data[self.label == 1][var]
            g0 = self.data[self.label == 0][var]
            if len(g1) > 0 and len(g0) > 0:
                u, _ = mannwhitneyu(g1, g0, alternative='two-sided')
                n1, n2 = len(g1), len(g0)
                wendt_a = 1 - ((2 * min(u, n1 * n2 - u)) / (n1 * n2))
                results.append({'Variable': var, 'Effect Size (A)': wendt_a})

        df = pd.DataFrame(results)

        if show_plot:
            plt.figure(figsize=(9, 5))
            plt.bar(df["Variable"], df["Effect Size (A)"], color="skyblue")
            plt.axhline(0.1, color="green", ls="--", lw=2)
            plt.axhline(0.3, color="#adac3c", ls="--", lw=2)
            plt.axhline(0.5, color="orange", ls="--", lw=2)
            plt.axhline(0.7, color="red", ls="--", lw=2)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Wendt's A")
            plt.title("Effect Size per Variable")
            plt.tight_layout()
            plt.show()

        return df.round(4)

    # ---------------------------------------------------------------------- #
    #  Spearman rank correlation (collinearity check)
    # ---------------------------------------------------------------------- #
    def spearman_correlation(self, threshold: float = 0.85, show_heatmap=True) -> pd.DataFrame:
        corr = self.data[self.numerical].corr(method='spearman')

        if show_heatmap:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f")
            plt.title("Spearman Rank Correlation Heatmap")
            plt.tight_layout()
            plt.show()

        high_corr = [
            (i, j, corr.loc[i, j])
            for i in corr.columns
            for j in corr.columns
            if i < j and abs(corr.loc[i, j]) > threshold
        ]
        redundant = pd.DataFrame(high_corr, columns=['Feature 1', 'Feature 2', 'Spearman ρ'])
        return redundant.sort_values('Spearman ρ', ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------------- #
    #  Summary Report
    # ---------------------------------------------------------------------- #
    def summary_report(self) -> pd.DataFrame:
        desc = self.descriptive_statistics()
        norm = self.shapiro_wilk().set_index('Variable')
        var = self.levene_test().set_index('Variable')
        eff = self.effect_size(show_plot=False).set_index('Variable')

        # Rename columns to avoid collisions
        norm = norm.rename(columns={'p-value': 'Normality p-value'})
        var = var.rename(columns={'p-value': 'Levene p-value'})

        summary = desc.join(
            [norm[['Normality p-value', 'Normality']],
             var[['Levene p-value', 'Equal Variance']],
             eff[['Effect Size (A)']]],
            how='left'
        )

        return summary.round(4)

