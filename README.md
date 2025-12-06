# 🫁 Tuberculosis Detection from Chest X‑ray Images

An end‑to‑end deep learning project to detect Tuberculosis (TB) from chest X‑ray images using a fine‑tuned VGG16 model and an interactive Streamlit web app for image upload and prediction.

---

## 🚀 Project Overview

Tuberculosis remains a major global health challenge, and chest X‑rays are a common, low‑cost screening tool.
This project uses transfer learning on a VGG16‑based convolutional neural network to classify chest X‑ray images as **TB** or **Normal**, then serves the model through a **Streamlit** web interface for easy experimentation and demonstration.

**Key features:**

- 🧠 Fine‑tuned VGG16 (binary classifier: TB vs Normal)  
- 📊 Training & validation on processed chest X‑ray dataset  
- 📈 Evaluation using accuracy, precision, recall, and F1‑score  
- 🌐 Streamlit web app with two pages: project intro + image prediction  
- 📁 Model exported as `.pth` and loaded in the app for fast inference  

> ⚠️ **Disclaimer:** This project is for educational and research purposes only and must **not** be used for real medical diagnosis.

---

## 🧠 Model & Training

- **Backbone:** VGG16 with the final fully‑connected layer replaced by a 2‑class output (TB, Normal). 
- **Input:** Preprocessed RGB chest X‑ray images resized to \(224 \times 224\) and normalized with ImageNet statistics.
- **Loss:** Binary / cross‑entropy loss.  
- **Metrics:** Accuracy, precision, recall, F1‑score on a held‑out validation or test set.  

The training pipeline (in `tb_detection_project.ipynb`) typically includes:

1. Loading and splitting the dataset into train/validation (and optionally test).  
2. Applying transforms: resize, normalization, and optional augmentation.  
3. Fine‑tuning the VGG16 model for several epochs.  
4. Saving the best checkpoint as `tb_vgg_model.pth` for deployment.

---

## 📊 Evaluation Metrics

After training, predictions on the validation/test dataloader are collected and evaluated using `scikit‑learn`:

- **Accuracy:** overall proportion of correctly classified images. 
- **Precision (TB):** of all images predicted as TB, how many are truly TB.  
- **Recall (TB):** of all true TB images, how many the model detects.  
- **F1‑score:** harmonic mean of precision and recall, useful for imbalanced datasets.  

----

## 🌐 Streamlit Web App

The Streamlit app (`app.py`) provides a simple two‑page UI:

1. **Introduction page**
   - Overview of the project and model.
   - Basic description and disclaimer.

2. **TB Prediction page**
   - Upload a chest X‑ray image (`.jpg`, `.jpeg`, `.png`).  
   - The app preprocesses the image and runs it through the loaded ResNet model.  
   - Displays:
     - Predicted label: **TB** or **Normal**  
     - Confidence score (softmax probability)  
     - Preview of the uploaded image  

The model is loaded once using a cached function so inference is fast and efficient.

---

## ▶️ How to Run the Project

### Option A – Run Streamlit app locally

From the project root:

streamlit run app.py

text
[web:40][web:99]

Then open the URL shown in the terminal (usually http://localhost:8501) in your browser.

### Option B – Run from Google Colab (for demo)

1. Open the Colab notebook (`notebooks/tb_detection.ipynb`).  
2. Upload `tb_resnet_final.pth` to the Colab runtime or mount Google Drive.  
3. Install dependencies and run:

!pip install streamlit torch torchvision pillow pyngrok

text

4. Start the app and expose via a tunnel (ngrok / Cloudflare, depending on setup), then open the generated public URL in your browser.

---

## 📌 Usage

1. Start the Streamlit app.  
2. Go to **🏠 Introduction** to read about the project.  
3. Switch to **🔬 TB Prediction** in the sidebar.  
4. Upload a chest X‑ray image.  
5. Wait for the model to run and view:
- Prediction: TB / Normal  
- Confidence score  
6. Experiment with multiple images and compare results.

---

## ⚠️ Medical Disclaimer

This repository is intended **only for learning, experimentation, and research.**  
It is **not** a certified medical device and must **not** be used to make clinical decisions or replace professional diagnosis. Always consult qualified healthcare professionals for any medical concerns.

---

## 🤝 Contributing

Contributions are welcome! Potential improvements:

- Trying different backbones (EfficientNet, DenseNet, etc.). 
- Better handling of class imbalance and calibration.  
- Explainability (Grad‑CAM, saliency maps) to highlight suspicious regions. 

Feel free to open issues or submit pull requests.

---

## 📧 Contact

If you have questions, suggestions, or feedback about this project, please open an issue in this repository or reach out via GitHub.

---

