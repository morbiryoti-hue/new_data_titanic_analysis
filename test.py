import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Survival rate analysis
print("\nOverall survival rate:", df['Survived'].mean())

# Survival by gender
print("\nSurvival rate by gender:")
print(df.groupby('Sex')['Survived'].mean())

# Survival by class
print("\nSurvival rate by passenger class:")
print(df.groupby('Pclass')['Survived'].mean())

# Age distribution
print("\nAge statistics:")
print(df['Age'].describe())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Survival count
sns.countplot(data=df, x='Survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival Count')
axes[0, 0].set_xticklabels(['Did not survive', 'Survived'])

# 2. Survival by gender
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].legend(['Did not survive', 'Survived'])

# 3. Survival by class
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[1, 0])
axes[1, 0].set_title('Survival by Passenger Class')
axes[1, 0].legend(['Did not survive', 'Survived'])

# 4. Age distribution
df['Age'].dropna().hist(bins=30, ax=axes[1, 1])
axes[1, 1].set_title('Age Distribution')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('titanic_analysis.png')
print("\nVisualization saved as 'titanic_analysis.png'")

# Additional insights
print("\nAverage age by survival status:")
print(df.groupby('Survived')['Age'].mean())

print("\nSurvival rate by embarkation port:")
print(df.groupby('Embarked')['Survived'].mean())
