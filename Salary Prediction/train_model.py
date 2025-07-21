# train_model.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("sample_salary_data.csv")

# Encode categorical columns
le_role = LabelEncoder()
le_edu = LabelEncoder()
le_loc = LabelEncoder()
le_size = LabelEncoder()

df["role"] = le_role.fit_transform(df["role"])
df["education"] = le_edu.fit_transform(df["education"])
df["location"] = le_loc.fit_transform(df["location"])
df["company_size"] = le_size.fit_transform(df["company_size"])

# Split features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump((le_role, le_edu, le_loc, le_size), "encoders.pkl")
