
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("MediBuddy Project Started Successfully!")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("MediBuddy Project Started Successfully!")

# Load Dataset 1
df1 = pd.read_excel("Medibuddy Insurance Data Price (1) (2).xlsx")

# Load Dataset 2
df2 = pd.read_excel("Medibuddy insurance data personal details (1) (2).xlsx")

# Display Dataset Information
print("\nDataset 1:")
print(df1.head())

print("\nDataset 2:")
print(df2.head())

# Display Columns
print("\nDataset 1 Columns:")
print(df1.columns.tolist())

print("\nDataset 2 Columns:")
print(df2.columns.tolist())

# Display Dataset Shape
print("\nDataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)


# Merge Two Datasets

df = pd.merge(
    df1,
    df2,
    on="Policy no.",
    how="inner",
    validate="one_to_one"
)

# Display Merged Dataset

print("\nMerged Dataset:")
print(df.head(10))

# Check Dataset Information

print("\nMerged Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# Check Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Merged Successfully!")


# Dataset Information

print("\nDataset Information:")
df.info()

# Statistical Summary

print("\nStatistical Summary:")
print(df.describe())

# Check Duplicate Values

print("\nDuplicate Records:")
print(df.duplicated().sum())

# Check Unique Values

print("\nUnique Values:")
print(df.nunique())

print("\nData Cleaning Check Completed!")


# Gender vs Insurance Charges Analysis

gender_analysis = df.groupby("sex")["charges in INR"].mean()

print("\nAverage Insurance Charges by Gender:")
print(gender_analysis)

# Bar Chart

gender_analysis.plot(
    kind="bar",
    figsize=(7, 5),
    color=["skyblue", "pink"]
)

plt.title("Average Insurance Charges by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Charges (INR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Question 2: Average Insurance Charges

average_charges = df["charges in INR"].mean()

print("\nAverage Insurance Charges:")
print("Rs.", round(average_charges, 2))

# Visualization

plt.figure(figsize=(7, 5))

plt.bar(
    ["Average Insurance Charges"],
    [average_charges]
)

plt.title("Average Amount Spent on Insurance")
plt.ylabel("Charges (INR)")

plt.tight_layout()
plt.show()

# Question 2: Average Insurance Charges

average_charges = df["charges in INR"].mean()

print("\nQuestion 2: Average Insurance Charges")
print("Average Amount: Rs.", round(average_charges, 2))



# Question 3: Region vs Insurance Charges

region_analysis = df.groupby("region")["charges in INR"].mean()

print("\nAverage Insurance Charges by Region:")
print(region_analysis)

# Visualization

plt.figure(figsize=(8, 5))

region_analysis.plot(kind="bar")

plt.title("Average Insurance Charges by Region")
plt.xlabel("Region")
plt.ylabel("Average Charges (INR)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Question 4: Children vs Insurance Charges

children_analysis = df.groupby("children")["charges in INR"].mean()

print("\nAverage Insurance Charges by Number of Children:")
print(children_analysis)

# Visualization

plt.figure(figsize=(8, 5))

children_analysis.plot(kind="bar")

plt.title("Insurance Charges vs Number of Children")
plt.xlabel("Number of Children")
plt.ylabel("Average Charges (INR)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Question 5: BMI vs Insurance Charges

correlation = df["bmi"].corr(df["charges in INR"])

print("\nBMI and Insurance Charges Correlation:")
print(round(correlation, 3))

# Scatter Plot

plt.figure(figsize=(8, 5))

plt.scatter(
    df["bmi"],
    df["charges in INR"],
    alpha=0.5
)

plt.title("BMI vs Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges (INR)")

plt.tight_layout()
plt.show()


# Question 6: Smoking vs Insurance Charges

smoker_analysis = df.groupby("smoker")["charges in INR"].mean()

print("\nAverage Insurance Charges by Smoking Status:")
print(smoker_analysis)

# Visualization

plt.figure(figsize=(7, 5))

smoker_analysis.plot(kind="bar", color=["skyblue", "salmon"])

plt.title("Smoker vs Non-Smoker Insurance Charges")
plt.xlabel("Smoking Status")
plt.ylabel("Average Charges (INR)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()



# Question 7: Age vs Insurance Charges

age_analysis = df.groupby("age")["charges in INR"].mean()

print("\nAverage Insurance Charges by Age:")
print(age_analysis)

# Visualization

plt.figure(figsize=(10, 5))

plt.plot(age_analysis.index, age_analysis.values, marker="o")

plt.title("Age vs Average Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Average Charges (INR)")
plt.grid(True)

plt.tight_layout()
plt.show()


# Question 8: Overall Insurance Analysis

print("\n===== OVERALL INSURANCE ANALYSIS =====")

print("\n1. Average Charges by Gender:")
print(df.groupby("sex")["charges in INR"].mean())

print("\n2. Average Charges by Region:")
print(df.groupby("region")["charges in INR"].mean())

print("\n3. Average Charges by Smoking Status:")
print(df.groupby("smoker")["charges in INR"].mean())

print("\n4. Average Charges by Age:")
print(df.groupby("age")["charges in INR"].mean())

print("\n5. Overall Average Insurance Charges:")
print(round(df["charges in INR"].mean(), 2))

print("\nOverall Analysis Completed Successfully!")


# BMI Based Discount Analysis

df["BMI_Category"] = pd.cut(
    df["bmi"],
    bins=[0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

bmi_analysis = df.groupby(
    "BMI_Category", observed=False
)["charges in INR"].mean()

print("\nBMI Based Insurance Charges:")
print(bmi_analysis)

bmi_analysis.plot(
    kind="bar",
    title="BMI Category vs Average Insurance Charges",
    xlabel="BMI Category",
    ylabel="Average Charges (INR)",
    rot=0
)

plt.tight_layout()
plt.show()



# Machine Learning - Insurance Charges Prediction

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Input and Output
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges in INR"]

# Separate numerical and categorical columns
categorical_columns = ["sex", "smoker", "region"]
numerical_columns = ["age", "bmi", "children"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("num", "passthrough", numerical_columns)
    ]
)

# Machine Learning Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\nMachine Learning Model Results")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
print("Actual Charges:", list(y_test.head()))
print("Predicted Charges:", list(y_pred[:5]))


from sklearn.model_selection import GridSearchCV


print("\n===== STARTING HYPERPARAMETER TUNING =====")

from sklearn.model_selection import GridSearchCV

param_grid = {
    "regressor__n_estimators": [100, 200],
    "regressor__max_depth": [None, 10, 20],
    "regressor__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("\n===== TUNING COMPLETED =====")
print("Best Parameters:", grid_search.best_params_)
print("Best CV R2 Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
tuned_predictions = best_model.predict(X_test)

print("Tuned MAE:",
      mean_absolute_error(y_test, tuned_predictions))

print("Tuned R2 Score:",
      r2_score(y_test, tuned_predictions))

