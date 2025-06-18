import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile

# Load model, scaler, and label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Feature extraction function
def extract_features(file_path, sr=16000, n_mfcc=40):
    y, _ = librosa.load(file_path, sr=sr)

    amplitude_envelope = np.max(np.abs(y))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)
    mel_std = np.std(mel_db, axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    features = np.concatenate((
        [amplitude_envelope, rms, zcr, centroid, bandwidth],
        mfcc_mean, mfcc_std,
        mel_mean, mel_std
    ))
    return features.reshape(1, -1)

# Streamlit UI
st.title("🎤 Speech Emotion Recognizer")
st.write("Upload a `.wav` file to predict the emotion expressed.")

uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        features = extract_features(tmp_file.name)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        predicted_emotion = label_encoder.inverse_transform(prediction)[0]

    st.audio(uploaded_file, format='audio/wav')
    st.success(f"🎯 **Predicted Emotion:** {predicted_emotion.upper()}")
