


# # new app.py

# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras.saving import load_model
# import traceback
# import matplotlib
# matplotlib.use('Agg')  # Use non-GUI backend for headless image generation
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# tf.config.set_visible_devices([], 'GPU')
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Ensure directories exist
# os.makedirs("uploads", exist_ok=True)
# os.makedirs("heatmaps", exist_ok=True)

# # Load Keras model
# model_path = "backend/models/real_fake_face_model.h5"
# try:
#     model = load_model("backend/models/real_fake_face_model.h5", compile=False)
#     # model = tf.keras.models.load_model(model_path, compile=False)
#     print("✅ Keras model loaded successfully!")
# except Exception as e:
#     print(f"❌ Failed to load model at {model_path}: {e}")
#     exit(1)

# # Preprocess image for CNN


# def preprocess(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError("Could not read image file.")
#     img = cv2.resize(img, (128, 128))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)
#     return img

# # Dummy heatmap generator (replace with Grad-CAM if needed)


# # def generate_heatmap(img_array):
# #     heatmap_path = os.path.join("heatmaps", "heatmap.png")
# #     # plt.imshow(img_array[0])
# #     plt.axis('off')
# #     plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
# #     plt.close()
# #     return heatmap_path

# def generate_heatmap(img_array):
#     import matplotlib.pyplot as plt  # safe after setting backend to 'Agg'
#     import os

#     heatmap_path = os.path.join("heatmaps", "heatmap.png")

#     # Use the 'Agg' backend to prevent GUI rendering issues
#     fig, ax = plt.subplots()
#     ax.imshow(img_array[0])
#     ax.axis('off')

#     fig.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
#     plt.close(fig)

#     return heatmap_path

# # Prediction route


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         if "image" not in request.files:
#             return jsonify({"error": "No image uploaded"}), 400

#         file = request.files["image"]
#         filename = file.filename
#         filepath = os.path.join("uploads", filename)
#         file.save(filepath)

#         img_array = preprocess(filepath)

#         # Predict probability
#         probability = model.predict(img_array)[0][0]  # Between 0 and 1

#         # Classify based on threshold
#         if probability < 0.4:
#             label = "Fake Image"
#         elif probability > 0.6:
#             label = "Real Image"
#         else:
#             label = "Uncertain"

#         # Direct confidence from model
#         confidence = round(probability * 100, 2)

#         heatmap_path = generate_heatmap(img_array)

#         return jsonify({
#             "label": label,
#             "confidence": confidence,
#             "heatmap": "/" + heatmap_path.replace("\\", "/")
#         })

#     except Exception as e:
#         print("❌ INTERNAL SERVER ERROR:")
#         traceback.print_exc()
#         return jsonify({"error": "INTERNAL SERVER ERROR"}), 500

# # Default route


# @app.route("/")
# def home():
#     return "Welcome to the Deepfake Detection API. Use /predict to POST an image."


# # Run server
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)


import os
import traceback
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
from keras.saving import load_model
import matplotlib

# Set backend to avoid GUI issues on servers
matplotlib.use('Agg')

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.config.set_visible_devices([], 'GPU')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create required directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)

# Load Keras model
MODEL_PATH = "backend/models/real_fake_face_model.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Image Preprocessing
def preprocess(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not read image file.")
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Heatmap Generator (Dummy Visualization)
def generate_heatmap(img_array):
    heatmap_path = os.path.join("heatmaps", "heatmap.png")
    fig, ax = plt.subplots()
    ax.imshow(img_array[0])
    ax.axis('off')
    fig.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return "/" + heatmap_path.replace("\\", "/")

# Prediction Endpoint
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
        probability = model.predict(img_array)[0][0]

        if probability < 0.4:
            label = "Fake Image"
        elif probability > 0.6:
            label = "Real Image"
        else:
            label = "Uncertain"

        confidence = round(probability * 100, 2)
        heatmap_path = generate_heatmap(img_array)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "heatmap": heatmap_path
        })

    except Exception as e:
        print("❌ INTERNAL SERVER ERROR:")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

# Welcome Route
@app.route("/")
def home():
    return "Welcome to the Deepfake Detection API. Use /predict to POST an image."

# Run Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
