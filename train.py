import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. Load Data
# Adjust the path if your file is in a different folder

df = pd.read_csv("data/car_data.csv")

# 2. Feature Engineering & Cleaning
# Create Brand column and drop Name
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# 3. Define Features (X) and Target (y)
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# 4. Apply Log Transformation to Target
# This was the key step that improved accuracy to 0.77
y_log = np.log(y)

# 5. Train-Test Split (Using the logged target)
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# 6. Build the Preprocessing Pipeline
num_col = ["year", "km_driven"]
cat_col = ["fuel", "seller_type", "brand", "transmission", "owner"]

# Pipeline for Numerical Columns
num_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scalar", StandardScaler())
])

# Pipeline for Categorical Columns
cat_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("oneencoder", OneHotEncoder(handle_unknown='ignore'))
])

# Combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_trans, num_col),
        ('cat', cat_trans, cat_col)
    ]
)

# 7. Create Final Pipeline with Linear Regression
pipe = Pipeline(steps=[
    ("column_transformer", preprocessor),
    ("model", LinearRegression())
])

# 8. Train the Model
print("Training model...")
pipe.fit(X_train, y_train_log)

# 9. Evaluate
# Predict on test data
y_pred_log = pipe.predict(X_test)

# Reverse the Log Transform (to get real currency values)
y_pred = np.exp(y_pred_log)
y_test_original = np.exp(y_test_log)

# Calculate Metrics
rmse = root_mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("------------------------------------------------")
print(f"Final RMSE: {rmse}")
print(f"Final R2 Score: {r2}")
print("------------------------------------------------")


joblib.dump(pipe, 'models/model.pkl')
print("Model saved successfully!")