from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image

# Auto detect: Render (Linux) → tflite-runtime, Windows → tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ModuleNotFoundError:
    import tensorflow as tf
    tflite = tf.lite

app = Flask(__name__)

# ⚠️ MUST match training class order
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMG_SIZE = 224

# Load TFLite model
interpreter = tflite.Interpreter(model_path="brain_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)

    # VGG16 preprocessing (manual)
    img = img[:, :, ::-1]      # RGB → BGR
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    img = Image.open(request.files['image']).convert("RGB")
    x = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(output))

    return jsonify({
        "class": CLASSES[idx],
        "confidence": round(float(output[idx]) * 100, 2)
    })

if __name__ == "__main__":
    app.run()
