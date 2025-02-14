from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained model
model = joblib.load("xgboost_email_spoofing_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Email Spoofing Detection API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({"spoofed": bool(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
