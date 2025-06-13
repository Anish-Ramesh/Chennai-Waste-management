# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load data and models
X = pd.read_csv("saved_models/X.csv")
y = pd.read_csv("saved_models/y.csv").iloc[:, 0]
model_performance = pd.read_csv("saved_models/performance.csv")
expected_columns = pd.read_csv("saved_models/columns.csv", header=None)[0].tolist()
zone_columns = [col for col in expected_columns if col.startswith("Zone_")]

models = {}
for filename in os.listdir("saved_models"):
    if filename.endswith(".pkl"):
        name = filename.replace(".pkl", "")
        model = joblib.load(os.path.join("saved_models", filename))
        models[name] = model

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        total = int(data["total_households"])
        covered = int(data["covered_households"])
        zone = data["zone_name"]

        coverage = covered / total * 100

        input_data = {
            "Total_Households": total,
            "Covered_Households": covered,
            "Coverage_Percentage": coverage
        }

        for col in zone_columns:
            input_data[col] = 1 if col == f"Zone_{zone}" else 0

        for col in expected_columns:
            if col not in input_data:
                input_data[col] = 0

        input_df = pd.DataFrame([input_data])[expected_columns]

        total_pred = 0
        for model in models.values():
            pred = float(model.predict(input_df)[0])
            pred = max(0, pred)
            total_pred += pred

        avg_pred = total_pred / len(models)
        print(f"HH_Source_Segregation = {avg_pred:.2f}")
        return jsonify({"HH_Source_Segregation": round(avg_pred, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
