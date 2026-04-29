from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("model/rf_model.pkl")

# Labels
labels = [
    'Anxiety','Blood_Pressure','Constipation','Cough','Cut',
    'Disease','Doctor','Emergency','Fever','Headache',
    'Infection','Nurse','Operation','Stomachache','Treatment'
]

# ================= ROOT TEST =================
@app.route("/")
def home():
    return "✅ Sign Language API is running on Render"

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if "landmarks" not in data:
            return jsonify({"error": "No landmarks found"}), 400

        landmarks = np.array(data["landmarks"], dtype=np.float32)

        # check size
        if landmarks.shape[0] != 225:
            return jsonify({"error": "Invalid landmark size"}), 400

        # reshape for model
        landmarks = landmarks.reshape(1, -1)

        # predict
        pred = model.predict(landmarks)[0]

        return jsonify({
            "prediction": labels[int(pred)]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= RUN SERVER =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)