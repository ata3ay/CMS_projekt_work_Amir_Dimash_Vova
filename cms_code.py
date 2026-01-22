import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from matplotlib.ticker import FuncFormatter  # for K formatting


# for loading data from the data we have
salary_data = pd.read_csv("DATA.csv")
print(salary_data.shape)
print(salary_data.columns)
print(salary_data.head())


# formatting into the form K for thousands
def format_thousands(value, position):
    return f"{int(value / 1000)}K"

def run_t_test(junior_salaries, senior_salaries):
    """
    H1: mean(senior) > mean(junior)
    t-test (equal_var=False) because variances can differ.
    """
    test_statistic, p_value = ttest_ind(
        senior_salaries,
        junior_salaries,
        equal_var=False,
        alternative="greater"
    )
    return test_statistic, p_value


def run_bootstrap_mean_difference(junior_salaries, senior_salaries, bootstrap_iterations=10000, random_seed=123):
    """
    Bootstraps the mean difference: mean(senior) - mean(junior)
    Returns:
      - bootstrap_differences (array)
      - confidence_interval_95 (tuple)
      - bootstrap_p_value (float): proportion of diffs <= 0
    """
    rng = np.random.default_rng(random_seed)

    bootstrap_differences = np.empty(bootstrap_iterations)

    for i in range(bootstrap_iterations):
        resampled_junior = rng.choice(junior_salaries, size=len(junior_salaries), replace=True)
        resampled_senior = rng.choice(senior_salaries, size=len(senior_salaries), replace=True)
        bootstrap_differences[i] = resampled_senior.mean() - resampled_junior.mean()

    ci_lower, ci_upper = np.percentile(bootstrap_differences, [2.5, 97.5])
    bootstrap_p_value = np.mean(bootstrap_differences <= 0)

    return bootstrap_differences, (ci_lower, ci_upper), bootstrap_p_value

def to_k(value):
    return f"{value/1000:.1f}K"

# plotting
def plot_salary_by_experience_group(data, selected_currency):

    # Filter and clean data 
    filtered_data = data[data["currency"] == selected_currency].copy()
    filtered_data = filtered_data.dropna(subset=["salary", "years_experience"])
    filtered_data = filtered_data[
        (filtered_data["salary"] > 0) &
        (filtered_data["years_experience"] >= 0)
    ]

    # Create experience groups for Specialists
    filtered_data["experience_group"] = np.where(
        filtered_data["years_experience"] <= 5,
        "0–5 years",
        ">5 years"
    )

    # Split into groups
    junior_employees = filtered_data[
        filtered_data["experience_group"] == "0–5 years"
    ].copy()

    senior_employees = filtered_data[
        filtered_data["experience_group"] == ">5 years"
    ].copy()

    # We need here salary arrays
    junior_salaries = junior_employees["salary"].to_numpy()
    senior_salaries = senior_employees["salary"].to_numpy()

    # validating that both groups have enough data
    if len(junior_salaries) < 5 or len(senior_salaries) < 5:
        print(f"\nNot enough data for reliable tests in {selected_currency}.")
        return

    # two-sample t-test
    t_test_statistic, t_test_p_value = run_t_test(junior_salaries, senior_salaries)

    mean_salary_junior = junior_salaries.mean()
    mean_salary_senior = senior_salaries.mean()
    mean_difference = mean_salary_senior - mean_salary_junior

    print("\nStatistical Tests")
    print(f"Currency: {selected_currency}")
    print(f"Average salary (0–5 years): {to_k(mean_salary_junior)}")
    print(f"Average salary (>5 years):  {to_k(mean_salary_senior)}")
    print(f"Average difference (>5 − 0–5): {to_k(mean_difference)}")

    print(f"t-test statistic: {t_test_statistic:.4f}")
    print(f"t-test p-value (H1: senior > junior): {t_test_p_value:.6f}")



    # Bootstrapping for average difference
    bootstrap_differences, bootstrap_ci_95, bootstrap_p_value = run_bootstrap_mean_difference(
        junior_salaries,
        senior_salaries,
        bootstrap_iterations=10000,
        random_seed=123
    )

    print(f"Bootstrap 95% CI for mean difference: ({bootstrap_ci_95[0]:.2f}, {bootstrap_ci_95[1]:.2f})")
    print(f" p-value: {bootstrap_p_value:.6f}")


    # for quick stats
    def print_basic_statistics(subset, group_name):
        print(f"\n{selected_currency} | {group_name}")
        print("Number of employees:", len(subset))
        print("Salary -> average:", subset["salary"].mean(),"| median:", subset["salary"].median())
        print("Experience -> average:", subset["years_experience"].mean(),"| median:", subset["years_experience"].median())


    print_basic_statistics(junior_employees, "0–5 years")
    print_basic_statistics(senior_employees, "> 5 years")

    # 0–5 years exp
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{selected_currency}: Employees with 0–5 years of experience",fontsize=14)

    axes[0].hist(
    junior_employees["salary"],
    bins=15,
    color="#2ecc71", 
    edgecolor="black",
    alpha=0.85
    )

    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({selected_currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))

    axes[1].hist(
    junior_employees["years_experience"], 
    bins=15,
    color="#2ecc71", 
    edgecolor="black",
    alpha=0.85)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    sns.stripplot(
        data=junior_employees,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.8,
        ax=axes[2],
        color="#27ae60",      
        edgecolor="black",
        linewidth=0.6
    )
    axes[2].set_title("Salary vs experience (jittered)")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({selected_currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(format_thousands))

    plt.tight_layout()
    plt.show()

    # >5 years exp
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{selected_currency}: Employees with more than 5 years of experience", fontsize=14)

    axes[0].hist(
    senior_employees["salary"],
    bins=15,
    color="#27ae60",  #green for money    
    edgecolor="black",
    alpha=0.85
    )

    axes[0].set_title("Salary distribution")
    axes[0].set_xlabel(f"Salary ({selected_currency})")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))

    axes[1].hist(senior_employees["years_experience"], bins=10,
    color="#27ae60",      
    edgecolor="black",
    alpha=0.85)
    axes[1].set_title("Experience distribution")
    axes[1].set_xlabel("Years of experience")
    axes[1].set_ylabel("Frequency")

    sns.stripplot(
        data=junior_employees,
        x="years_experience",
        y="salary",
        jitter=0.25,
        alpha=0.8,
        ax=axes[2],
        color="#27ae60",      
        edgecolor="black",
        linewidth=0.6
        )
    axes[2].set_title("Salary vs experience")
    axes[2].set_xlabel("Years of experience")
    axes[2].set_ylabel(f"Salary ({selected_currency})")
    axes[2].yaxis.set_major_formatter(FuncFormatter(format_thousands))

    plt.tight_layout()
    plt.show()

    # Bootstrap distribution plot
    plt.figure(figsize=(8, 4))

    plt.hist(bootstrap_differences, bins=40, alpha=0.8,
    color="#27ae60",     
    edgecolor="black",
    )

    # Observed mean difference
    plt.axvline(
        mean_difference,
        color="black",
        linewidth=2,
        label="Observed difference (average)"
    )

    # Zero line (no difference)
    plt.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=2,
        label="No difference (0)"
    )

    # 95% confidence interval
    plt.axvline(
        bootstrap_ci_95[0],
        linestyle="--",
        linewidth=2,
        label="95% CI low"
    )

    plt.axvline(
        bootstrap_ci_95[1],
        linestyle="--",
        linewidth=2,
        label="95% CI high"
    )

    plt.title(f"Bootstrap distribution of average salary difference ({selected_currency})")
    plt.xlabel(f"Difference in averages ({selected_currency})  (>5 years − 0–5 years)")
    plt.ylabel("Bootstrap frequency")

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_thousands))
    plt.legend()
    plt.tight_layout()
    plt.show()



# We have used both currencies here
plot_salary_by_experience_group(salary_data, "EUR")
plot_salary_by_experience_group(salary_data, "USD")
