import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# =====================================================
# 1. Load & Preprocess Dataset
# =====================================================
def load_and_preprocess_data():
    data = pd.read_csv('Data.csv')
    data.columns = data.columns.str.strip()

    # KEEP Ward No. exactly as-is
    data = data.rename(columns={
        'Total No. of households / establishments': 'Total_Households',
        'Total no. of households and establishments covered through doorstep collection': 'Covered_Households',
        'HH covered with Source Seggeratation': 'HH_Source_Segregation',
        'Zone Name': 'Zone_Name'
    })

    # Ensure Ward No. exists
    if 'Ward No.' not in data.columns:
        raise Exception("Column 'Ward No.' not found in Data.csv. Check exact spelling!")

    # Remove invalid rows
    data = data[data['Total_Households'] > 0].fillna(0)
    return data

# =====================================================
# 2. Train & Save Model
# =====================================================
def train_model():
    data = load_and_preprocess_data()

    # Label encode Zone_Name
    le = LabelEncoder()
    data['Zone_ID'] = le.fit_transform(data['Zone_Name'])
    joblib.dump(le, 'saved_models/zone_encoder.pkl')

    # --- Include Ward No. as-is ---
    X = data[['Total_Households', 'Covered_Households', 'Zone_ID', 'Ward No.']]
    y = data['HH_Source_Segregation']

    os.makedirs("saved_models", exist_ok=True)
    pd.Series(X.columns.tolist()).to_csv("saved_models/columns.csv", index=False, header=False)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    print("\nTraining XGBoost with Ward No. included...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"Train R²: {r2_score(y_train, y_pred_train):.4f} | "
          f"Test R²: {r2_score(y_test, y_pred_test):.4f} | "
          f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_test) * 100:.2f}%")

    joblib.dump(model, "saved_models/XGBoost.pkl")
    print("\nModel saved successfully!")
    return model

# =====================================================
# 3. Prediction Function (Ward No. preserved)
# =====================================================
def predict_segregation(input_data, model_name='XGBoost'):
    try:
        model = joblib.load(f"saved_models/{model_name}.pkl")
        le = joblib.load('saved_models/zone_encoder.pkl')
        columns = pd.read_csv("saved_models/columns.csv", header=None)[0].tolist()
    except FileNotFoundError:
        raise Exception("Model or encoder not found. Train the model first.")

    zone_name = input_data.get('Zone_Name', '')
    zone_id = le.transform([zone_name])[0] if zone_name in le.classes_ else 0

    # --- Keep Ward No. EXACTLY ---
    input_df = pd.DataFrame([{
        'Total_Households': input_data['Total_Households'],
        'Covered_Households': input_data['Covered_Households'],
        'Zone_ID': zone_id,
        'Ward No.': input_data['Ward No.']
    }])

    input_df = input_df.reindex(columns=columns, fill_value=0)

    pred = model.predict(input_df)[0]
    return round(min(max(0, pred), input_data['Covered_Households']))  # clamp to valid range

# =====================================================
# 4. Run Training & Example Prediction
# =====================================================
if __name__ == "__main__":
    os.makedirs("saved_models", exist_ok=True)
    model = train_model()

    example_input = {
        'Total_Households': 10000,
        'Covered_Households': 3443,
        'Zone_Name': 'Thiruvotriyur',
        'Ward No.': 1     # ← EXACT column name
    }

    print("\nExample Prediction:")
    print(predict_segregation(example_input))