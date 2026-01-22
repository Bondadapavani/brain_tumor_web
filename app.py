from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "brain_tumor_vgg16_final.h5"
)
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""
    image_file = None

    if request.method == 'POST':
        file = request.files['file']

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = img.reshape(1, 224, 224, 3)

            # Predict
            preds = model.predict(img)
            prediction = class_labels[np.argmax(preds)]

            image_file = file.filename

    return render_template(
        'index.html',
        prediction=prediction,
        image_file=image_file
    )

# 🔥 NEW ROUTE TO SERVE UPLOADED IMAGES
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



