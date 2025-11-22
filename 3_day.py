# ---------------------------------------------
# Linear & Logistic Regression using Iris Dataset
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# =============================================
# LOAD IRIS DATASET
# =============================================

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("First 5 rows of dataset:\n")
print(df.head())
print("\n=====================================\n")


# =============================================
# 1. LINEAR REGRESSION
# Predict petal length from sepal length
# =============================================

print("LINEAR REGRESSION\n")

X_linear = df[['sepal length (cm)']]
y_linear = df['petal length (cm)']

X_train, X_test, y_train, y_test = train_test_split(
    X_linear, y_linear, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("Slope (Coefficient):", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)


# Plot Linear Regression
plt.figure()
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred_linear, label="Regression Line")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Linear Regression - Iris Dataset")
plt.legend()
plt.show()


# =============================================
# 2. LOGISTIC REGRESSION
# Classify Setosa vs Others
# =============================================

print("\nLOGISTIC REGRESSION\n")

# Convert to binary classification
df['is_setosa'] = (df['species'] == 0).astype(int)

X_logistic = df[['sepal length (cm)', 'sepal width (cm)']]
y_logistic = df['is_setosa']

X_train, X_test, y_train, y_test = train_test_split(
    X_logistic, y_logistic, test_size=0.2, random_state=42
)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_log)
print("Logistic Regression Accuracy:", accuracy)


# =============================================
# PREDICTION EXAMPLES
# =============================================

print("\nSAMPLE PREDICTIONS\n")

# Linear Prediction
sample_sepal_length = [[5.5]]
predicted_petal = linear_model.predict(sample_sepal_length)
print("Predicted petal length for sepal length 5.5 cm:", predicted_petal[0])

# Logistic Prediction
sample_flower = [[5.1, 3.5]]
species_prediction = log_model.predict(sample_flower)
print("Is Setosa? (1 = Yes, 0 = No):", species_prediction[0])
