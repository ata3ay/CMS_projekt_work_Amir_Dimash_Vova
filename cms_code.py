import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter #for K with numbers


df = pd.read_csv("DATA.csv")
print(df.shape)
print(df.columns)
print(df.head())

def thousands_formatter(x, pos):
    return f"{int(x/1000)}K"


def plots_by_experience_groups(df, currency):
    # 1) Filter + clean
    d = df[df["currency"] == currency].copy()
    d = d.dropna(subset=["salary", "years_experience"])
    d = d[(d["salary"] > 0) & (d["years_experience"] >= 0)]

    # 2) Create groups
    d["exp_group"] = np.where(d["years_experience"] <= 5, "0-5", ">5")

    # 3) Split
    d_low = d[d["exp_group"] == "0-5"].copy()
    d_high = d[d["exp_group"] == ">5"].copy()

    # Helper to show basic stats
    def quick_stats(sub, name):
        print(f"\n--- {currency} | Group {name} ---")
        print("N =", len(sub))
        print("Salary: mean =", sub["salary"].mean(), "median =", sub["salary"].median())
        print("Experience: mean =", sub["years_experience"].mean(), "median =", sub["years_experience"].median())

    quick_stats(d_low, "0-5")
    quick_stats(d_high, ">5")

    # ===== 0–5 years =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{currency}: Employees with 0–5 years of experience", fontsize=14)

    axes[0].hist(d_low["salary"], bins=25)
    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(thousands_formatter)) ##X achse 



    axes[1].hist(d_low["years_experience"], bins=10)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    # jittered salary vs experience (stripplot)
    sns.stripplot(
        data=d_low,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.7,
        ax=axes[2]
    )
    axes[2].set_title("Salary vs experience (jittered)")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


    plt.tight_layout()
    plt.show()

    # ===== >5 years =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{currency}: Employees with >5 years of experience", fontsize=14)

    axes[0].hist(d_high["salary"], bins=25)
    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(thousands_formatter)) ##X achse 

    

    axes[1].hist(d_high["years_experience"], bins=10)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    sns.stripplot(
        data=d_high,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.7,
        ax=axes[2]
    )
    axes[2].set_title("Salary vs experience (jittered)")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


    plt.tight_layout()
    plt.show()


plots_by_experience_groups(df, "EUR")
plots_by_experience_groups(df, "USD")
