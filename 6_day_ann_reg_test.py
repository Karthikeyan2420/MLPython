import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = load_model("regression_model.h5")

# ---- Example test row ----
new_employee = pd.DataFrame([{
    "emp_id": 3,
    "age": 40,
    "department": "Finance",
    "education": "Masters",
    "job_role": "Manager",
    "years_at_company": 6,
    "monthly_income": 30000,
    "distance_from_home": 20,
    "num_projects": 5,
    "avg_monthly_hours": 250,
    "performance_score": 80
}])

# ---- Apply preprocessing ----
X_new = preprocessor.transform(new_employee)

# ---- Predict ----
prediction = model.predict(X_new)
print("Predicted left value:", float(prediction[0]))
