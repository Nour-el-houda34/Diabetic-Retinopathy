from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model(r"C:\Users\pc\Downloads\best_model_diabetic.keras", compile=False)
classes = ['Healthy', 'Mild DR', 'Moderate DR', 'Proliferate DR', 'Severe DR']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = ""
    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Préparer l'image
            img = Image.open(image_path).resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prédiction
            pred = model.predict(img_array)
            pred_class = classes[np.argmax(pred)]
            confidence = np.max(pred) * 100
            result = f"Classe prédite : {pred_class} ({confidence:.2f}%)"
    return render_template("index.html", result=result, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
