import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

# ---- Step 1: Load CSV ----
df = pd.read_csv("employee_data_ann.csv")

# ---- Target ----
target = "left"
X = df.drop(columns=[target])
y = df[target]

# ---- Identify numeric + categorical ----
numeric_cols = ["emp_id", "age", "years_at_company", "monthly_income",
                "distance_from_home", "num_projects", "avg_monthly_hours",
                "performance_score"]

categorical_cols = ["department", "education", "job_role"]

# ---- Preprocessor (scaling + OHE) ----
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

X_processed = preprocessor.fit_transform(X)
joblib.dump(preprocessor, "preprocessor.pkl")

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# ---- Reshape for LSTM ----
# LSTM expects 3D data: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# ---- Build RNN Model ----
model = Sequential([
    LSTM(64, activation="tanh", return_sequences=False, input_shape=(1, X_train.shape[2])),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)   # Regression output
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---- Train ----
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# ---- Save model ----
model.save("rnn_regression_model.h5")

print("RNN model + preprocessor saved!")
