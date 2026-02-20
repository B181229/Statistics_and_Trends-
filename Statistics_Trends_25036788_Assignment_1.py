#!/usr/bin/env python
# coding: utf-8

# Statistics_Trends-Assignment
# Student_Name: Sukesh Kumar Eddagiri
# Student_ID: 25036788
# Dataset: StudentsPerformance.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis


# Loading the dataset
df = pd.read_csv("StudentsPerformance.csv")

# 1. RELATIONAL_PLOT (Scatter Plot)

plt.figure()
sns.scatterplot(data=df, x="reading score", y="writing score", hue="gender")
plt.title("Reading Score vs Writing Score")
plt.xlabel("Reading Score")
plt.ylabel("Writing Score")
plt.savefig("Relational_Plot.png")
plt.tight_layout()
plt.show()


# 2. CATEGORICAL_PLOT (Bar Plot)

avg_math = df.groupby("gender")["math score"].mean().reset_index()

plt.figure()
sns.barplot(data=avg_math, x="gender", y="math score")
plt.title("Average Math Score by Gender")
plt.savefig("Categorical_Plot.png")
plt.tight_layout()
plt.show()

# 3. STATISTICAL_PLOT (Correlation Heatmap)

scores_cols = ["math score", "reading score", "writing score"]
cor = df[scores_cols].corr()

plt.figure()
sns.heatmap(cor, annot=True, cmap="coolwarm", fmt=".3f")
plt.title("Correlation Heatmap of Scores")
plt.savefig("Statistical_Plot.png")
plt.tight_layout()
plt.show()

# 4. FOUR_STATISTICAL_MOMENTS (Math Score)

math_score = df["math score"]

mean_value = np.mean(math_score)
variance_value = np.var(math_score)
skewness_value = skew(math_score)
kurtosis_value = kurtosis(math_score)

print("Statistical Moments for Math Score:")
print("Mean:", mean_value)
print("Variance:", variance_value)
print("Skewness:", skewness_value)
print("Kurtosis:", kurtosis_value)





