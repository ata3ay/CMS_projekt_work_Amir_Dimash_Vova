# CMS Project: Salary Analysis in AI and Data Science

## Project Idea and Research Question
This project analyses the salaries of professionals working in the fields of Artificial Intelligence and Data Science.
The main research question is:

**Do professionals with more than five years of experience earn more than those with less experience (0–5 years)?**

The goal is to investigate whether work experience has a statistically significant effect on salary differences.

---

## Dataset
The dataset is taken from Kaggle:

**AI and Data Science Job Salaries (2020–2025)**  
https://www.kaggle.com/datasets/pratyushmishradev/ai-and-data-science-job-salaries-20202025

The dataset contains information about:
- salary
- currency
- years of professional experience
- job-related attributes

---

## Data Pre-processing
To avoid bias caused by different currencies, the analysis is restricted to a single currency (USD).
The data is cleaned by:
- removing missing values in salary and experience
- keeping only positive salary values
- grouping professionals into two experience categories:
  - **0–5 years**
  - **More than 5 years**

---

## Methods
The following statistical methods are used:

- **Two-sample t-test**  
  to test whether the mean salaries of the two experience groups differ significantly.

- **Bootstrapping**  
  to estimate confidence intervals for the difference in mean salaries and to validate the robustness of the results without relying on normality assumptions.

---

## Evaluation
The results are evaluated using:
- p-values from the t-test
- bootstrap confidence intervals
- visual inspection of the bootstrap sampling distribution

A statistically significant result is concluded if:
- the p-value is below 0.05
- the bootstrap confidence interval does not include zero

---

## Project Structure
CMS_Projekt/
│── cms_code.py
│── DATA.csv
│── README.md


---

## How to Run the Code
1. Install required Python packages:
```bash
pip install pandas numpy scipy matplotlib seaborn
```
2. Run the analysis script:
```
python cms_code.py
```

## Authors

Dinmukhamed Atabay
Amir Kazken 
Vladimir Ovcharov