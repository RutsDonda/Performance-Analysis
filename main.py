#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------- Q1 Load Dataset --------------------
df = pd.read_csv("FINAL Data MSD.csv")

df.drop(columns=['Ground', 'Date', 'Match_ID', 'Opposition'], inplace=True)

# Replace DNB with NaN
df.replace('DNB', np.nan, inplace=True)

print(df)

# -------------------- Helper Function --------------------
def count_nan(dataframe):
    return dataframe.isna().sum()

print("NaN Count Before Preprocessing:\n", count_nan(df))

# -------------------- Q2 Handle Missing Values --------------------
num_cols = ['Runs', 'Inns', 'Position', 'BF', '4s', '6s', 'SR']
cat_cols = ['Dismissal']

# Mean per class label (Match_Type)
for col in num_cols:
    df[col] = df.groupby('Match_Type')[col].transform(
        lambda x: x.astype(float).fillna(x.astype(float).mean())
    )

# Mode per class label
for col in cat_cols:
    df[col] = df.groupby('Match_Type')[col].transform(
        lambda x: x.fillna(x.mode()[0])
    )

print("NaN Count After Preprocessing:\n", count_nan(df))

# -------------------- Q3 Statistical Analysis --------------------
print("\nStatistical Summary:")
print(df[num_cols].describe())

print("\nMode of Match_Type:")
print(df['Match_Type'].mode())

# -------------------- Q4 Unique Values --------------------
for column in df.columns:
    print(f"\nColumn: {column}")
    print("Unique Values:", df[column].unique())
    print("Value Counts:\n", df[column].value_counts())

# -------------------- Q5 Visualization --------------------
numeric_columns = ['Runs', 'BF', '4s', '6s', 'SR']

color_map = {'ODI': 'r', 'Test': 'g', 'T20': 'b'}
colors = df['Match_Type'].map(color_map)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_columns):
    axes[i].scatter(df[col], df['SR'], c=colors, s=8)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Strike Rate")

plt.tight_layout()
plt.show()

# -------------------- Q6 KNN Classification --------------------
X = df[numeric_columns]
Y = df['Match_Type']

# Encode target labels
le = LabelEncoder()
Y = le.fit_transform(Y)

# Feature scaling (important for KNN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

best_k = 0
best_accuracy = 0

for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(Y_test, preds)

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

print("\nBest K:", best_k)
print("Best Accuracy:", best_accuracy * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, knn.predict(X_test)))
