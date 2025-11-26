import numpy as np
import pandas as pd

def generate_employee_data(n_samples=3000):
    rng = np.random.default_rng(42)

    age = rng.integers(20, 60, size=n_samples)
    years_at_company = rng.integers(0, 25, size=n_samples)
    monthly_income = rng.normal(50000, 15000, size=n_samples).clip(12000, 180000).astype(int)
    distance_from_home = rng.integers(1, 60, size=n_samples)
    num_projects = rng.integers(1, 10, size=n_samples)
    avg_monthly_hours = rng.integers(90, 300, size=n_samples)

    departments = ['Sales', 'HR', 'Engineering', 'Finance', 'Marketing', 'Support']
    education_levels = ['HighSchool', 'Bachelors', 'Masters', 'PhD']
    job_roles = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']

    department = rng.choice(departments, size=n_samples)
    education = rng.choice(education_levels, size=n_samples)
    job_role = rng.choice(job_roles, size=n_samples)

    performance_score = rng.integers(30, 100, size=n_samples)

    # Attrition target: 0 = stayed, 1 = left
    attrition_prob = (
        0.1 +
        0.2 * (distance_from_home / 60) +
        0.2 * (1 - performance_score / 100)
    )
    left = (rng.random(n_samples) < attrition_prob).astype(int)

    df = pd.DataFrame({
        "emp_id": np.arange(1, n_samples + 1),
        "age": age,
        "department": department,
        "education": education,
        "job_role": job_role,
        "years_at_company": years_at_company,
        "monthly_income": monthly_income,
        "distance_from_home": distance_from_home,
        "num_projects": num_projects,
        "avg_monthly_hours": avg_monthly_hours,
        "performance_score": performance_score,
        "left": left
    })

    return df

# Generate and save
df = generate_employee_data(30000)
df.to_csv("employee_data_ann.csv", index=False)

print("✅ CSV file generated successfully: employee_data_ann.csv")
print(df.head())
