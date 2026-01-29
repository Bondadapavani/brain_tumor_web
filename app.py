import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = load_model("brain_tumor_vgg16_final.h5")
classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- MRI VALIDATION FUNCTION --------
def is_mri_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Check color variance (MRI images are low-color)
    b, g, r = cv2.split(img)
    color_variance = np.var(b) + np.var(g) + np.var(r)

    # Edge detection (MRI has strong structures)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Decision thresholds (tested values)
    if color_variance < 5000 and edge_density > 0.02:
        return True
    else:
        return False

# -------- FLASK ROUTE --------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # STEP 1: MRI CHECK
            if not is_mri_image(file_path):
                result = "Invalid Image (Please upload MRI scan)"
                confidence = None
                return render_template("index.html", result=result, confidence=confidence)

            # STEP 2: CNN PREDICTION
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)
            max_prob = np.max(pred)
            class_index = np.argmax(pred)

            confidence = round(float(max_prob) * 100, 2)

            # FINAL SAFETY CHECK
            if max_prob < 0.85:
                result = "Uncertain MRI Image"
            else:
                result = classes[class_index]

    return render_template("index.html", result=result, confidence=confidence)

if __name__ == "__main__":
    app.run()
