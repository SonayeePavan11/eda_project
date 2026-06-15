import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# 1. LOAD DATASET (SAFE VERSION)
# ==============================

file_path = r"C:\Users\pavan\OneDrive\Desktop\Theranax\eda_project\data.csv"

# Check file exists before loading
if not os.path.exists(file_path):
    print("❌ ERROR: data.csv not found!")
    print("👉 Check file path:", file_path)
    exit()

df = pd.read_csv(file_path)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

# ==============================
# 2. BASIC INFORMATION
# ==============================
print("\n--- DATA INFO ---")
print(df.info())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# ==============================
# 3. MISSING VALUES CHECK
# ==============================
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Fill missing values (numeric only)
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# ==============================
# 4. UNIVARIATE ANALYSIS
# ==============================
print("\n--- UNIVARIATE ANALYSIS ---")

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ==============================
# 5. CATEGORICAL ANALYSIS
# ==============================
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    plt.figure()
    sns.countplot(y=df[col])
    plt.title(f"Count Plot of {col}")
    plt.show()

# ==============================
# 6. BIVARIATE ANALYSIS
# ==============================
print("\n--- BIVARIATE ANALYSIS ---")

if len(num_cols) >= 2:
    sns.pairplot(df[num_cols])
    plt.show()

# ==============================
# 7. CORRELATION HEATMAP
# ==============================
print("\n--- CORRELATION MATRIX ---")

plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# 8. OUTLIERS DETECTION
# ==============================
print("\n--- OUTLIERS CHECK ---")

for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ==============================
# 9. KEY INSIGHTS
# ==============================
print("\n--- KEY INSIGHTS ---")

for col in num_cols:
    print(f"{col}: Mean={df[col].mean():.2f}, Max={df[col].max()}, Min={df[col].min()}")

print("\n✅ EDA Completed Successfully!")