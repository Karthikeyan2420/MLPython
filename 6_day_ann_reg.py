import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ---- Step 1: Load CSV ----
df = pd.read_csv("employee_data_ann.csv")

# ---- Identify feature and target columns ----
target = "left"
X = df.drop(columns=[target])
y = df[target]

# ---- Separate numeric and categorical columns ----
numeric_cols = ["emp_id", "age", "years_at_company", "monthly_income",
                "distance_from_home", "num_projects", "avg_monthly_hours",
                "performance_score"]

categorical_cols = ["department", "education", "job_role"]

# ---- Preprocess: OneHotEncode categorical + scale numeric ----
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# ---- Transform the features ----
X_processed = preprocessor.fit_transform(X)

# Save the preprocessor so you can use it later
joblib.dump(preprocessor, "preprocessor.pkl")

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# ---- ANN Model ----
model = Sequential([
    Dense(64, activation="relu", input_dim=X_train.shape[1]),
    Dense(32, activation="relu"),
    Dense(1)   # Regression: no activation
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---- Train ----
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# ---- Save model ----
model.save("regression_model.h5")

print("Model + preprocessor saved!")
