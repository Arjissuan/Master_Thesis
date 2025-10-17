from src import StatisticalModule, LData

LD = LData()
# LD.drop_na()
data, labels =LD.features_df, LD.labels()['antigram+']   # example loading step

stats_module = StatisticalModule(data, labels)

# 1. Basic summary
desc = stats_module.descriptive_statistics()
print(desc)

# 2. Normality test
normality = stats_module.shapiro_wilk()

# 3. Variance equality
levene = stats_module.levene_test()

# 4. Multiple testing correction
mtc = stats_module.multiple_testing_correction()

# 5. Effect size
effect = stats_module.effect_size()

# 6. Spearman redundancy
redundant = stats_module.spearman_correlation(threshold=0.85)

# 7. Combined summary report
summary = stats_module.summary_report()
print(summary)
