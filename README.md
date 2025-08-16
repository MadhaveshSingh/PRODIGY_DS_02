# PRODIGY_DS_02

Titanic Data Cleaning & Exploratory Data Analysis (EDA)

This project focuses on performing data cleaning and exploratory data analysis (EDA) on the famous Titanic dataset. The goal is to preprocess the dataset, handle missing values, and explore patterns that influenced passenger survival.

🔹 Steps Performed

**1. Data Inspection**

Loaded the dataset and checked the first few rows.
Inspected dataset information, data types, and summary statistics.
Identified missing values in important columns.


**2. Data Cleaning**

Filled missing Age values with the median.
Replaced missing Embarked values with the mode.
Dropped the Cabin column due to excessive missing values.


**3. Exploratory Data Analysis (EDA)**

Visualized key insights using Matplotlib and Seaborn:
Survival Count: Distribution of survivors vs. non-survivors.
Survival by Gender: Women had a significantly higher survival rate.
Survival by Passenger Class (Pclass): First-class passengers had higher survival chances compared to third-class.
Age Distribution: Age spread of passengers with a peak among younger individuals.
Fare Distribution by Class: Higher fares were associated with upper-class passengers.
Correlation Heatmap: Explored correlations between numerical features.


**4. Statistical Insights**

Average survival rate by gender.
Average survival rate by passenger class.
Average age comparison between survivors and non-survivors.


📊 Key Findings

Gender played a major role – females had higher survival rates.
Class mattered – first-class passengers had better chances of survival.
Younger passengers had slightly better survival chances compared to older ones.

⚙️ Technologies Used

Python
Pandas (Data cleaning & preprocessing)
Matplotlib & Seaborn (Data visualization)
