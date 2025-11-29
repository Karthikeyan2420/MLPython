import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------------
# 1. Load CSV
# -------------------------------------------------------
data = pd.read_csv("employee_data_ann.csv")   # <--- change to your file

# -------------------------------------------------------
# 2. Separate features and target
# -------------------------------------------------------
X = data.drop("left", axis=1)
y = data["left"]

# -------------------------------------------------------
# 3. Identify categorical and numeric columns
# -------------------------------------------------------
categorical_cols = ["department", "education", "job_role"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# -------------------------------------------------------
# 4. Preprocessing (OneHot + Scaling)
# -------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# -------------------------------------------------------
# 5. Split data
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 6. Fit preprocessing
# -------------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to numpy arrays for TensorFlow
X_train_processed = np.array(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed)
X_test_processed = np.array(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed)

# -------------------------------------------------------
# 7. Build ANN Model
# -------------------------------------------------------
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(X_train_processed.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary output
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------------
# 8. Train Model
# -------------------------------------------------------
history = model.fit(
    X_train_processed,
    y_train,
    epochs=25,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# -------------------------------------------------------
# 9. Evaluate Model
# -------------------------------------------------------
y_pred = (model.predict(X_test_processed) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
