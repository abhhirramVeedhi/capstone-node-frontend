# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import io

# app = Flask(__name__)
# CORS(app)  # Allow frontend requests

# # Load the trained model
# model_path = "backend/models/real_fake_face_model.pkl"
# try:
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
#     print("✅ Model loaded successfully!")
# except FileNotFoundError:
#     print("❌ Model file not found! Ensure 'real_fake_face_model.pkl' exists.")
#     exit(1)

# # Image preprocessing function
# def preprocess_image(img):
#     img = img.convert("RGB")  # Ensure 3-channel input
#     img = img.resize((128, 128))  # Resize to match model input
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize pixel values
#     return img_array

# @app.route("/")
# def home():
#     return "Welcome to the Fake Image Detection API! Use /predict to upload an image."

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     img = Image.open(io.BytesIO(file.read()))  # Read image from request
#     img_array = preprocess_image(img)

#     # Get prediction
#     prediction = model.predict(img_array)[0][0]  # Extract probability score
#     probability = float(prediction)  # Convert to standard float

#     # Debugging info
#     print(f"🔍 Model Prediction Score: {probability}")

#     # **Improved Threshold Logic**
# if probability < 0.4:
#     result = "Fake Image"
# elif probability > 0.6:
#     result = "Real Image"
# else:
#     result = "Uncertain"  # Middle-range values may indicate poor confidence

#     return jsonify({"prediction": result, "confidence": probability})

# if __name__ == "__main__":
#     app.run(debug=True)


# new app.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.saving import load_model
import traceback
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless image generation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.config.set_visible_devices([], 'GPU')
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)

# Load Keras model
model_path = "backend/models/real_fake_face_model.h5"
try:
    model = load_model("backend/models/real_fake_face_model.h5", compile=False)
    # model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Keras model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model at {model_path}: {e}")
    exit(1)

# Preprocess image for CNN


def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image file.")
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Dummy heatmap generator (replace with Grad-CAM if needed)


# def generate_heatmap(img_array):
#     heatmap_path = os.path.join("heatmaps", "heatmap.png")
#     # plt.imshow(img_array[0])
#     plt.axis('off')
#     plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
#     plt.close()
#     return heatmap_path

def generate_heatmap(img_array):
    import matplotlib.pyplot as plt  # safe after setting backend to 'Agg'
    import os

    heatmap_path = os.path.join("heatmaps", "heatmap.png")

    # Use the 'Agg' backend to prevent GUI rendering issues
    fig, ax = plt.subplots()
    ax.imshow(img_array[0])
    ax.axis('off')

    fig.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return heatmap_path

# Prediction route


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        filename = file.filename
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        img_array = preprocess(filepath)

        # Predict probability
        probability = model.predict(img_array)[0][0]  # Between 0 and 1

        # Classify based on threshold
        if probability < 0.4:
            label = "Fake Image"
        elif probability > 0.6:
            label = "Real Image"
        else:
            label = "Uncertain"

        # Direct confidence from model
        confidence = round(probability * 100, 2)

        heatmap_path = generate_heatmap(img_array)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "heatmap": "/" + heatmap_path.replace("\\", "/")
        })

    except Exception as e:
        print("❌ INTERNAL SERVER ERROR:")
        traceback.print_exc()
        return jsonify({"error": "INTERNAL SERVER ERROR"}), 500

# Default route


@app.route("/")
def home():
    return "Welcome to the Deepfake Detection API. Use /predict to POST an image."


# Run server
if __name__ == "__main__":
    app.run(port=5000, debug=True)


# import os
# import traceback
# from flask import Flask, request, jsonify
# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
# import numpy as np

# app = Flask(__name__)

# # ---- MODEL LOADING ----

# def load_keras_model(model_path):
#     try:
#         # Workaround for InputLayer deserialization issue
#         with tf.keras.utils.custom_object_scope({}):
#             model = tf.keras.models.load_model(model_path, compile=False)
#         print("✅ Keras model loaded successfully!")
#         return model
#     except Exception as e:
#         print(f"❌ Failed to load model at {model_path}: {e}")
#         traceback.print_exc()
#         exit(1)

# model_path = "backend/models/real_fake_face_model.h5"
# model = load_keras_model(model_path)

# # ---- IMAGE PREDICTION ----

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(128, 128))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor = img_tensor / 255.0  # normalize
#     return img_tensor

# @app.route("/upload", methods=["POST"])
# def upload_file():
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         file_path = os.path.join("uploads", file.filename)
#         os.makedirs("uploads", exist_ok=True)
#         file.save(file_path)

#         # Preprocess and predict
#         img_tensor = preprocess_image(file_path)
#         prediction = model.predict(img_tensor)
#         predicted_class = "Fake" if prediction[0][0] > 0.5 else "Real"

#         return jsonify({
#             "filename": file.filename,
#             "prediction": predicted_class,
#             "confidence": float(prediction[0][0])
#         })
#     except Exception as e:
#         print("🔥 INTERNAL ERROR:", str(e))
#         traceback.print_exc()
#         return jsonify({"error": "Internal Server Error"}), 500

# # ---- MAIN ----

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
