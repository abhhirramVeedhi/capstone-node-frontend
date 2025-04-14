# 🧠 DeepFake Detection using CNN and Xception Networks

## 📌 Overview

This project focuses on detecting deepfake videos and images using deep learning techniques. It uses Convolutional Neural Networks (CNNs) and the Xception architecture to distinguish between real and manipulated facial media. Deepfakes are a growing concern in digital content manipulation, and this project addresses the challenge with a trained model and visual analysis pipeline.

## 🎯 Objectives

- Identify deepfake images using a trained Xception model.
- Use benchmark datasets like FaceForensics++ for robust training.
- Analyze model performance using accuracy, loss graphs, confusion matrix, and ROC curve.
- Provide an end-to-end notebook for data handling, model training, and evaluation.

## 🛠 Tech Stack

- **Language**: Python
- **Frameworks/Libraries**:
  - TensorFlow / Keras
  - OpenCV
  - NumPy, Pandas
  - Matplotlib, Seaborn
- **Model**: Xception Network (pretrained on ImageNet)
- **Notebook Environment**: Jupyter Notebook

## 📁 Project Structure
deepfake-detection/
│
├── check3.ipynb              # Main Jupyter Notebook with full model pipeline
├── dataset/                  # Folder containing training and test images
│   ├── real/                 # Real face images
│   └── fake/                 # Deepfake face images
│
├── models/                   # Saved model weights or checkpoints
│   └── xception_model.h5     # Trained Xception model
│
├── results/                  # Output graphs and evaluation metrics
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── capstone.zip              # Zipped backup of entire project (optional)





## 📊 Dataset

- **FaceForensics++** (compressed image dataset): Provides labeled real and fake media for training and testing.
- Data preprocessing includes:
  - Frame extraction from videos (if applicable)
  - Image resizing to 256x256
  - Data augmentation (rotation, flip, brightness adjustments)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook check3.ipynb
Follow the steps inside the notebook to train, test, and visualize the model.

✅ Results
Model Accuracy: Achieved ~90% test accuracy

Loss Function: Binary cross-entropy

Evaluation Tools:

Confusion matrix

ROC-AUC score

Visual predictions

✨ Features
Deepfake classification using transfer learning (Xception)

Frame-by-frame prediction compatibility

One-time review submission logic (future scope)

Clean, modular training pipeline

Easily extendable to other architectures like EfficientNet, ResNet

📚 References
[1] FaceForensics++ Dataset – https://github.com/ondyari/FaceForensics

[2] Xception Paper – https://arxiv.org/abs/1610.02357

[3] Keras Xception Documentation – https://keras.io/api/applications/xception/

[4] CNN Visual Guide – https://cs231n.github.io/convolutional-networks/

[5] TensorFlow Docs – https://www.tensorflow.org/

[6] Adam Optimizer – https://arxiv.org/abs/1412.6980

[7] ROC Curves Explained – https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

👨‍💻 Author
Name: Abhhirram Veedhi

Institution: Vellore Institute of Technology 

Project Type: Capstone / Final Year Project

Developed On: Local system using Jupyter Notebook

⚠️ Note: This project currently runs locally and is not deployed. Future work may include building a UI, enabling video stream classification, or deploying on a cloud-based service for real-time inference.
