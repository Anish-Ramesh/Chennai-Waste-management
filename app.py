from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
import os
import json
from Wastepredictor import predict_segregation, train_model

app = Flask(__name__)
CORS(app)

# Load model performance if available
try:
    model_performance = pd.read_csv("saved_models/performance.csv").to_dict('records')
    print("Loaded model performance metrics")
except:
    model_performance = []

# Load zones from data for mapping numeric codes -> zone names
try:
    _zones_df = pd.read_csv("Data.csv")
    _zones_df.columns = _zones_df.columns.str.strip()
    _unique_zones = list(dict.fromkeys(_zones_df['Zone Name'].astype(str).str.strip().tolist()))
except Exception:
    _unique_zones = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Validate input
        required_fields = ['total_households', 'covered_households']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        try:
            total = int(data["total_households"])
            covered = int(data["covered_households"])
            zone_number = data.get("zone_number") or data.get("zone")
            zone_name = data.get("zone_name")
            
            if total <= 0:
                return jsonify({"error": "Total households must be greater than 0"}), 400
                
            if covered < 0 or covered > total:
                return jsonify({"error": "Covered households must be between 0 and total households"}), 400
                
        except ValueError as e:
            return jsonify({"error": "Invalid input values. Please provide valid numbers."}), 400

        # Determine Zone_Name for the model
        zone_name_for_model = None
        if isinstance(zone_name, str) and zone_name.strip():
            zone_name_for_model = zone_name.strip()
        elif zone_number is not None and _unique_zones:
            try:
                idx = int(zone_number) - 1
                if 0 <= idx < len(_unique_zones):
                    zone_name_for_model = _unique_zones[idx]
            except ValueError:
                zone_name_for_model = None
        # Fallback: use first zone if nothing provided
        if zone_name_for_model is None and _unique_zones:
            zone_name_for_model = _unique_zones[0]

        # Infer a representative Ward No. for this zone from Data.csv
        ward_no_for_model = 1
        try:
            if zone_name_for_model and not _zones_df.empty:
                _zones_df_local = _zones_df.copy()
                _zones_df_local.columns = _zones_df_local.columns.str.strip()
                mask = _zones_df_local['Zone Name'].astype(str).str.strip() == zone_name_for_model
                candidates = _zones_df_local[mask]
                if not candidates.empty and 'Ward No.' in candidates.columns:
                    ward_no_for_model = int(candidates['Ward No.'].iloc[0])
        except Exception:
            ward_no_for_model = 1

        # Prepare input for prediction (model uses zone name and ward number)
        input_data = {
            'Total_Households': total,
            'Covered_Households': covered,
            'Zone_Name': zone_name_for_model,
            'Ward No.': ward_no_for_model
        }
        
        try:
            # Get prediction (absolute segregated households)
            pred_count = predict_segregation(input_data)

            # Derive rate from predicted count
            segregation_rate = round((pred_count / total) * 100, 2) if total > 0 else 0.0

            # Prepare response
            response = {
                'prediction': {
                    'segregation_rate': segregation_rate,
                    'predicted_households': int(pred_count),
                    'model_used': 'XGBoost'
                },
                'input': {
                    'total_households': total,
                    'covered_households': covered,
                    'coverage_percentage': round((covered / total) * 100, 2),
                    'zone_number': zone_number,
                    'zone_name': zone_name_for_model,
                    'ward_no': ward_no_for_model
                },
                'model_performance': model_performance
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "error": "Prediction failed",
                "details": str(e)
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500

@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        # You might want to add authentication here in production
        print("Retraining model...")
        train_model()

        return jsonify({
            "status": "success",
            "message": "Model retrained successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Model training failed",
            "error": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": len(model_performance) > 0
    })


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Return aggregated waste segregation stats per zone for the dashboard."""
    try:
        df = pd.read_csv("Data.csv")
        df.columns = df.columns.str.strip()

        # Basic renaming for clarity
        df = df.rename(columns={
            'Total No. of households / establishments': 'Total_Households',
            'Total no. of households and establishments covered through doorstep collection': 'Covered_Households',
            'HH covered with Source Seggeratation': 'HH_Source_Segregation',
            'Zone Name': 'Zone_Name'
        })

        df = df[df['Total_Households'] > 0]

        # Aggregate by zone
        zone_group = df.groupby('Zone_Name', as_index=False).agg({
            'Total_Households': 'sum',
            'Covered_Households': 'sum',
            'HH_Source_Segregation': 'sum'
        })

        zone_group["Coverage_Rate"] = (zone_group["Covered_Households"] / zone_group["Total_Households"] * 100).round(2)
        zone_group["Segregation_Rate"] = (zone_group["HH_Source_Segregation"] / zone_group["Total_Households"] * 100).round(2)

        # City totals
        city_totals = {
            "Total_Households": int(zone_group["Total_Households"].sum()),
            "Covered_Households": int(zone_group["Covered_Households"].sum()),
            "HH_Source_Segregation": int(zone_group["HH_Source_Segregation"].sum()),
        }
        city_totals["Coverage_Rate"] = round(city_totals["Covered_Households"] / city_totals["Total_Households"] * 100, 2)
        city_totals["Segregation_Rate"] = round(city_totals["HH_Source_Segregation"] / city_totals["Total_Households"] * 100, 2)

        return jsonify({
            "zones": zone_group.to_dict(orient="records"),
            "city_totals": city_totals
        })
    except Exception as e:
        return jsonify({
            "error": "Failed to load dashboard data",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # If no model exists, train it
    if not os.path.exists("saved_models/XGBoost.pkl"):
        print("No trained model found. Training new model...")
        try:
            train_model()
            print("Model trained successfully!")
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
