
from flask import Flask, request, jsonify
import model     # import from model.py
import json

app = Flask(__name__)

# Train the model once at startup
trained_model = model.train_model()

@app.route("/", methods=["GET"])
def home():
    return "Google Stock Prophet Model Deployed Successfully!"

@app.route("/predict", methods=["POST"])
def predict():
    req = request.json
    days = req.get("days", 30)  # default = 30 days prediction

    forecast = model.make_prediction(trained_model, days)

    # Convert forecast dataframe to JSON
    return jsonify(forecast.to_dict(orient="records"))

if __name__ == "__main__":
    app.run()
