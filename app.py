from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "pneumonia_resnet_finetuned 2.keras"  # adjust path if needed
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ["Normal", "Pneumonia"]

import requests
from io import BytesIO
from PIL import Image

def preprocess_image(img_source, target_size=(224,224)):
    if img_source.startswith("http"):  # Handle URLs
        response = requests.get(img_source)
        img = Image.open(BytesIO(response.content)).convert("RGB")  # 🔑 Force RGB
        img = img.resize(target_size)
    else:  # Local path
        img = Image.open(img_source).convert("RGB")  # 🔑 Force RGB
        img = img.resize(target_size)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Pneumonia Prediction API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    demo_url = request.form.get("demo_url")

    if file:
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)
        img_array = preprocess_image(file_path, target_size=(224,224))
        os.remove(file_path)
    elif demo_url:
        img_array = preprocess_image(demo_url, target_size=(224,224))
    else:
        return jsonify({"error": "No image provided"}), 400

    prediction = model.predict(img_array)[0][0]
    predicted_class = CLASS_NAMES[int(prediction > 0.5)]
    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    })



# Endpoint to return list of demo images

from flask import url_for
@app.route("/demo-images", methods=["GET"])
def demo_images():
    demo_folder = os.path.join(app.static_folder, "demo_images")
    files = os.listdir(demo_folder)
    image_urls = [url_for("static", filename=f"demo_images/{f}", _external=True) for f in files]
    return jsonify({"images": image_urls})


if __name__ == "__main__":
    app.run(debug=True)
