import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if not exists
os.makedirs("output", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

if os.path.exists("Titanic.csv"):
    df = pd.read_csv("Titanic.csv")
elif os.path.exists("data/Titanic.csv"):
    df = pd.read_csv("data/Titanic.csv")
else:
    print("ERROR: Titanic.csv not found!")
    print("Place Titanic.csv in:")
    print("1. Project folder OR")
    print("2. data folder")
    exit()

print("\nDataset Loaded Successfully!")

# -----------------------------
# Basic Information
# -----------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

if 'Cabin' in df.columns:
    df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Statistical Summary
# -----------------------------

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# Survival Count
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.savefig('output/survival_count.png')
plt.close()

# -----------------------------
# Gender Distribution
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title('Gender Distribution')
plt.savefig('output/gender_distribution.png')
plt.close()

# -----------------------------
# Survival by Gender
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.savefig('output/survival_by_gender.png')
plt.close()

# -----------------------------
# Passenger Class
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.savefig('output/passenger_class.png')
plt.close()

# -----------------------------
# Age Distribution
# -----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('output/age_distribution.png')
plt.close()

# -----------------------------
# Fare Distribution
# -----------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.savefig('output/fare_distribution.png')
plt.close()

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title('Correlation Heatmap')
plt.savefig('output/correlation_heatmap.png')
plt.close()

# -----------------------------
# Boxplot of Age
# -----------------------------

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Age'])
plt.title('Age Boxplot')
plt.savefig('output/age_boxplot.png')
plt.close()

# -----------------------------
# Boxplot of Fare
# -----------------------------

plt.figure(figsize=(6,4))
sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot')
plt.savefig('output/fare_boxplot.png')
plt.close()

# -----------------------------
# Survival by Passenger Class
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')
plt.savefig('output/survival_by_class.png')
plt.close()

# -----------------------------
# Age vs Fare Scatter Plot
# -----------------------------

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='Age',
    y='Fare',
    hue='Survived',
    data=df
)
plt.title('Age vs Fare')
plt.savefig('output/age_vs_fare.png')
plt.close()

# -----------------------------
# Pairplot
# -----------------------------

pairplot_data = df[['Age', 'Fare', 'Pclass', 'Survived']]
sns.pairplot(pairplot_data)
plt.savefig('output/pairplot.png')
plt.close()

# -----------------------------
# Findings
# -----------------------------

print("\nEDA COMPLETED SUCCESSFULLY")
print("\nCharts saved in OUTPUT folder.")

print("\nKey Findings:")
print("1. Female passengers had a higher survival rate.")
print("2. First-class passengers survived more often.")
print("3. Most passengers were between 20 and 40 years old.")
print("4. Higher fare passengers generally had better survival chances.")
print("5. Dataset contains some outliers in Fare.")