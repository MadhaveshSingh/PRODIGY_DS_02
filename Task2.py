# Titanic Data Cleaning & EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/Madhavesh/Downloads/train.csv")


# 1. DATA INSPECTION

print("First 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# 2. DATA CLEANING


# Fill missing 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop 'Cabin' due to too many missing values
df = df.drop(columns=['Cabin'])


# 3. EDA


# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count')
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.show()

# Survival by Passenger class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set3')
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Fare distribution by class
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set2')
plt.title('Fare Distribution by Class')
plt.show()

# Correlation heatmap (numerical features)
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


print("\nAverage survival rate by gender:")
print(df.groupby('Sex')['Survived'].mean())

print("\nAverage survival rate by class:")
print(df.groupby('Pclass')['Survived'].mean())

print("\nAverage age by survival status:")
print(df.groupby('Survived')['Age'].mean())