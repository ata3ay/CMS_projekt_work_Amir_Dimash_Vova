import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter  # for K formatting


# ===== Load data =====
salary_data = pd.read_csv("DATA.csv")

print(salary_data.shape)
print(salary_data.columns)
print(salary_data.head())


# ===== Formatter for thousands (e.g. 50K) =====
def format_thousands(value, position):
    return f"{int(value / 1000)}K"


# ===== Main plotting function =====
def plot_salary_by_experience_group(data, selected_currency):

    # 1) Filter and clean data
    filtered_data = data[data["currency"] == selected_currency].copy()
    filtered_data = filtered_data.dropna(subset=["salary", "years_experience"])
    filtered_data = filtered_data[
        (filtered_data["salary"] > 0) &
        (filtered_data["years_experience"] >= 0)
    ]

    # 2) Create experience groups
    filtered_data["experience_group"] = np.where(
        filtered_data["years_experience"] <= 5,
        "0–5 years",
        ">5 years"
    )

    # 3) Split into groups
    junior_employees = filtered_data[
        filtered_data["experience_group"] == "0–5 years"
    ].copy()

    senior_employees = filtered_data[
        filtered_data["experience_group"] == ">5 years"
    ].copy()

    # ===== Helper function for quick statistics =====
    def print_basic_statistics(subset, group_name):
        print(f"\n--- {selected_currency} | {group_name} ---")
        print("Number of employees:", len(subset))
        print(
            "Salary → mean:",
            subset["salary"].mean(),
            "| median:",
            subset["salary"].median()
        )
        print(
            "Experience → mean:",
            subset["years_experience"].mean(),
            "| median:",
            subset["years_experience"].median()
        )

    print_basic_statistics(junior_employees, "0–5 years")
    print_basic_statistics(senior_employees, ">5 years")

    # ===== 0–5 years group =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{selected_currency}: Employees with 0–5 years of experience",
        fontsize=14
    )

    axes[0].hist(junior_employees["salary"], bins=25)
    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({selected_currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))

    axes[1].hist(junior_employees["years_experience"], bins=10)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    sns.stripplot(
        data=junior_employees,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.7,
        ax=axes[2]
    )
    axes[2].set_title("Salary vs experience (jittered)")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({selected_currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(format_thousands))

    plt.tight_layout()
    plt.show()

    # ===== >5 years group =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{selected_currency}: Employees with more than 5 years of experience",
        fontsize=14
    )

    axes[0].hist(senior_employees["salary"], bins=25)
    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({selected_currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))

    axes[1].hist(senior_employees["years_experience"], bins=10)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    sns.stripplot(
        data=senior_employees,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.7,
        ax=axes[2]
    )
    axes[2].set_title("Salary vs experience (jittered)")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({selected_currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(format_thousands))

    plt.tight_layout()
    plt.show()


# ===== Run for both currencies =====
plot_salary_by_experience_group(salary_data, "EUR")
plot_salary_by_experience_group(salary_data, "USD")
