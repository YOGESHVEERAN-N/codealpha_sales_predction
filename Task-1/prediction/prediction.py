# ==========================================
# SALES PREDICTION USING LINEAR REGRESSION
# ==========================================

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 50)
print("SALES PREDICTION USING PYTHON")
print("=" * 50)

# ------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------

print("\nLoading Dataset...")

df = pd.read_csv("Advertising.csv")

print("\nFirst 5 Rows:")
print(df.head())

# ------------------------------------------
# STEP 2: DATA CLEANING
# ------------------------------------------

print("\nCleaning Data...")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ------------------------------------------
# STEP 3: DATA EXPLORATION
# ------------------------------------------

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------------------
# STEP 4: CORRELATION ANALYSIS
# ------------------------------------------

print("\nGenerating Correlation Heatmap...")

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ------------------------------------------
# STEP 5: VISUALIZATION
# ------------------------------------------

plt.figure(figsize=(6, 4))
sns.scatterplot(x="TV", y="Sales", data=df)
plt.title("TV Advertising vs Sales")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="Radio", y="Sales", data=df)
plt.title("Radio Advertising vs Sales")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="Newspaper", y="Sales", data=df)
plt.title("Newspaper Advertising vs Sales")
plt.show()

# ------------------------------------------
# STEP 6: FEATURE SELECTION
# ------------------------------------------

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# ------------------------------------------
# STEP 7: TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ------------------------------------------
# STEP 8: TRAIN MODEL
# ------------------------------------------

print("\nTraining Linear Regression Model...")

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------------
# STEP 9: PREDICTIONS
# ------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------
# STEP 10: MODEL EVALUATION
# ------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------
# STEP 11: ACTUAL VS PREDICTED
# ------------------------------------------

results = pd.DataFrame({
    "Actual Sales": y_test,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales")
print(results.head(10))

# ------------------------------------------
# STEP 12: FEATURE IMPORTANCE
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Importance")
print(importance.sort_values(by="Coefficient", ascending=False))

# ------------------------------------------
# STEP 13: PLOT ACTUAL VS PREDICTED
# ------------------------------------------

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# ------------------------------------------
# STEP 14: FUTURE SALES PREDICTION
# ------------------------------------------

print("\nFuture Sales Prediction")
print("-" * 30)

tv = float(input("Enter TV Advertising Budget: "))
radio = float(input("Enter Radio Advertising Budget: "))
newspaper = float(input("Enter Newspaper Advertising Budget: "))

future_data = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

future_sales = model.predict(future_data)

print("\nPredicted Future Sales = {:.2f}".format(future_sales[0]))

# ------------------------------------------
# STEP 15: BUSINESS INSIGHTS
# ------------------------------------------

print("\nBUSINESS INSIGHTS")
print("-" * 30)

best_feature = importance.loc[
    importance["Coefficient"].idxmax(),
    "Feature"
]

print(f"Most Influential Advertising Channel: {best_feature}")
print("TV advertising has the strongest impact on sales.")
print("Radio advertising also contributes positively.")
print("Newspaper advertising has the least influence.")
print("Businesses should invest more in high-performing channels.")

print("\nProject Completed Successfully!")