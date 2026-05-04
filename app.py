# Import Flask (used to create API)
from flask import Flask, request, jsonify

import numpy as np          # numerical operations
import cv2                 # image processing (OpenCV)

from keras.models import load_model   # load trained model


# Create Flask app
app = Flask(__name__)


# -----------------------------
# 📦 LOAD TRAINED MODEL
# -----------------------------

# Load the model you trained earlier
model = load_model("facialemotionmodel.h5")


# Emotion labels (must match training order)
emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Neutral", "Sad", "Surprise"
]


# -----------------------------
# 🖼️ IMAGE PREPROCESS FUNCTION
# -----------------------------

def preprocess_image(img):
    # Resize image to 48x48 (same as training)
    img = cv2.resize(img, (48, 48))

    # Convert to grayscale (1 channel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values (0–255 → 0–1)
    img = img / 255.0

    # Reshape to match model input shape
    # (batch_size, height, width, channels)
    img = np.reshape(img, (1, 48, 48, 1))

    return img


# -----------------------------
# 🚀 PREDICTION API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():
    # Get image file sent from Flutter
    file = request.files["image"]

    # Convert file → OpenCV image
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    # Preprocess image
    processed = preprocess_image(img)

    # Make prediction
    prediction = model.predict(processed)

    # Get index of highest probability
    emotion_index = np.argmax(prediction)

    # Convert index → label
    emotion = emotion_labels[emotion_index]

    # Send result back as JSON
    return jsonify({
        "emotion": emotion,
        "confidence": float(np.max(prediction))
    })


# -----------------------------
# ▶️ RUN SERVER
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
