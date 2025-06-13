# wastepredictor.py
import pandas as pd 
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load dataset
data = pd.read_csv(r'C:\Users\Anish\OneDrive\Desktop\Anish Python Projects 1\Wastemanage\Data.csv')
data.columns = data.columns.str.strip()

# Drop unnecessary columns
columns_to_drop = [col for col in ['Ward Name', 'City Name'] if col in data.columns]
data = data.drop(columns=columns_to_drop)

# Rename columns
data = data.rename(columns={
    'Total No. of households / establishments': 'Total_Households',
    'Total no. of households and establishments covered through doorstep collection': 'Covered_Households',
    'HH covered with Source Seggeratation': 'HH_Source_Segregation',
    'Zone Name': 'Zone_Name'
})

if 'HH_Source_Segregation' not in data.columns:
    raise ValueError("Column 'HH_Source_Segregation' not found after renaming.")

data = data.fillna(0)

# Feature engineering
data['Coverage_Percentage'] = (data['Covered_Households'] / data['Total_Households']) * 100

# Encode categorical
data = pd.get_dummies(data, columns=['Zone_Name'], prefix='Zone', drop_first=False)

# Save column list
column_list = data.drop(columns=['HH_Source_Segregation']).columns.tolist()
pd.Series(column_list).to_csv("saved_models/columns.csv", index=False, header=False)

X = data.drop(columns=['HH_Source_Segregation'])
y = data['HH_Source_Segregation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LinearRegression': LinearRegression(),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
}

os.makedirs("saved_models", exist_ok=True)
model_performance = []
all_preds = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    joblib.dump(model, f"saved_models/{name}.pkl")

    y_pred = model.predict(X)
    y_pred = np.maximum(y_pred, 0)  # Clamp negatives to 0
    all_preds[name] = y_pred

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    model_performance.append({'Model': name, 'MSE': mse, 'R2': r2})
    print(f"{name} saved. R2: {r2:.2f}, MSE: {mse:.2f}")

# Save average predictions
df_all_preds = pd.DataFrame(all_preds)
df_all_preds['Average_Prediction'] = df_all_preds.mean(axis=1)
df_all_preds['Actual'] = y.values
df_all_preds.to_csv("saved_models/final_predictions.csv", index=False)

# Save performance and inputs
X.to_csv("saved_models/X.csv", index=False)
y.to_csv("saved_models/y.csv", index=False)
pd.DataFrame(model_performance).to_csv("saved_models/performance.csv", index=False)

print("HH_Source_Segregation prediction values saved.")
