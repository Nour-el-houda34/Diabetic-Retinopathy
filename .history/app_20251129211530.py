from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# --- CHEMINS ---
UPLOAD_FOLDER = "static/uploads"
HISTORY_FILE = "history.txt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CLASSES DU MODÈLE ---
classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# --- CHARGEMENT DU MODÈLE ---
MODEL_PATH = "C:\\Users\\pc\\Downloads\\best_model_diabetic.keras"
model = load_model(MODEL_PATH)


# --- GESTION HISTORIQUE ---
def add_history(filename, result):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {filename} | {result}\n")


def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip().split(" | ") for line in f.readlines()]


# --- ROUTE PRINCIPALE ---
@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    result = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]

            if file.filename != "":
                filename = file.filename  # On garde seulement le nom
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)

                # Prétraitement
                img = Image.open(save_path).convert("RGB").resize((224, 224))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Prédiction
                pred = model.predict(img_array)
                pred_class = classes[np.argmax(pred)]
                confidence = np.max(pred) * 100
                result = f"{pred_class} ({confidence:.2f}%)"

                # Historique
                add_history(filename, result)

                image_path = filename  # <-- CORRECTION IMPORTANTE

    history = read_history()

    return render_template("index.html", image_path=image_path, result=result, history=history)


if __name__ == "__main__":
    app.run(debug=True)
