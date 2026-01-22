import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np

df = pd.read_csv("DATA.csv")
print(df.shape)
print(df.columns)
df.head()

df["currency"].value_counts().head(10)

df1 = df[df["currency"] == "EUR"].copy()
print(df1.shape)

df1 = df1.dropna(subset=["salary", "years_experience"])
df1 = df1[(df1["salary"] > 0) & (df1["years_experience"] >= 0)]
print(df1.shape)

df1["exp_group"] = df1["years_experience"].apply(lambda x: "0-5" if x <= 5 else ">5")
df1["exp_group"].value_counts()

summary = df1.groupby("exp_group")["salary"].agg(
    n="count", mean="mean", median="median", std="std"
)
print(summary)

# 5.2 Boxplot – visual check
sns.boxplot(data=df1, x="exp_group", y="salary")
plt.title("Salary by experience group (USD)")
plt.show()

low = df1[df1["exp_group"] == "0-5"]["salary"].to_numpy()
high = df1[df1["exp_group"] == ">5"]["salary"].to_numpy()

t_stat, p_val = ttest_ind(high, low, equal_var=False, alternative="greater")
print("Welch t-test:")
print("t =", t_stat)
print("p =", p_val)
print("mean diff (high-low) =", high.mean() - low.mean())

rng = np.random.default_rng(123)
B = 10000

boot_diff = np.empty(B)
n_low, n_high = len(low), len(high)

for i in range(B):
    s_low = rng.choice(low, size=n_low, replace=True)
    s_high = rng.choice(high, size=n_high, replace=True)
    boot_diff[i] = s_high.mean() - s_low.mean()

# 95% CI (percentile)
ci_low, ci_high = np.percentile(boot_diff, [2.5, 97.5])
boot_p = np.mean(boot_diff <= 0)   # для H1: diff > 0

print("Bootstrap:")
print("95% CI =", (ci_low, ci_high))
print("bootstrap p-value =", boot_p)

plt.hist(boot_diff, bins=50)
plt.axvline(0, linestyle="--")
plt.title("Bootstrap distribution: mean salary difference (>5 - 0-5)")
plt.xlabel("Difference")
plt.ylabel("Frequency")
plt.show()



