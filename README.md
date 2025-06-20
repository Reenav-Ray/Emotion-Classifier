# 🎤 Emotion Classification on Speech Data

This project focuses on the classification of human emotions using speech recordings. By leveraging signal processing techniques and machine learning models, we classify emotions such as *Angry, Happy, Sad, Calm, Fearful*, and more from `.wav` audio files.

---

## 📌 Project Description

The goal of this project is to build a fully functional, accurate emotion recognition system that works on real-world audio data. The solution involves extracting meaningful features from raw `.wav` files, training a machine learning classifier to recognize emotion patterns, and deploying the model as a user-friendly web application using Streamlit.

---

## 🛠️ Preprocessing Methodology

We applied the following preprocessing steps to the audio data:

1. **Standardization of audio sampling rate** (16,000 Hz).
2. **Silence trimming and normalization** handled during `librosa.load`.
3. **Feature Extraction**:
   - **MFCCs** (Mel Frequency Cepstral Coefficients)
   - **Delta & Delta-Delta MFCCs**
   - **Mel Spectrogram (converted to dB scale)**
   - **Spectral Centroid** (brightness)
   - **Spectral Bandwidth**
   - **RMS Energy** (signal power)
   - **Zero Crossing Rate**
   - **Amplitude Envelope**

All extracted features were flattened into fixed-length vectors using mean and standard deviation operations.

4. **Feature Scaling**:
   - Standardized using `StandardScaler` to zero mean and unit variance.

5. **Label Extraction**:
   - Emotion labels were extracted from file names following RAVDESS-style naming convention.

---

## 🧠 Model Pipeline

The machine learning pipeline consists of the following components:

- **Input**: `.wav` audio file
- **Feature Extraction**: `librosa`-based feature extractor
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Classifier**: `XGBoost` model (`XGBClassifier`)
- **Validation Strategy**: Stratified 80/20 split to preserve label distribution
- **Output**: Predicted emotion label (e.g., “Angry”)

> Models and encoders are saved using `joblib` and loaded into the deployment interface.

---

## 📊 Accuracy Metrics

- ✅ **Validation Accuracy**: **97.48%**
- ✅ **Weighted F1 Score**: High across all classes
- ✅ **Validation Strategy**: Stratified Shuffle Split (80% train, 20% validation)

Confusion matrix and classification report confirm robust performance across all emotion classes.

---

## 🌐 Deployment

The model is deployed using [**Streamlit**](https://streamlit.io/) as a web app that:

- Accepts `.wav` file uploads
- Extracts features in real-time
- Uses the trained model to predict emotion
- Outputs prediction and plays uploaded audio

To run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
