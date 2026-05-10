import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ------------------------------
# App Config
# ------------------------------
st.set_page_config(page_title="Cat Sound Classifier", layout="centered")
st.title("🐱 Cat Sound Classification App")

# ------------------------------
# Load Model & Label Binarizer
# ------------------------------
MODEL_PATH = "cat_sound_cnn_model.h5"   
model = tf.keras.models.load_model(MODEL_PATH)

label_classes = ["Angry", "Defence", "Fighting", "Happy", "HuntingMind", "Mating", "MotherCall", "Paining", "Resting", "Warning"]

# ------------------------------
# Audio Preprocessing
# ------------------------------
SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_FRAMES = 96

def extract_mel_2d(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, max_frames=MAX_FRAMES):
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), 'constant')

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.T

    if mel_db.shape[0] < max_frames:
        mel_db = np.pad(mel_db, ((0, max_frames - mel_db.shape[0]), (0, 0)), mode='constant')
    else:
        mel_db = mel_db[:max_frames, :]

    return mel_db[..., np.newaxis]  


def predict_audio(file):
    # Load audio
    y, sr = librosa.load(file, sr=SR)

    # Extract mel
    mel = extract_mel_2d(y)

    # Expand batch dimension
    mel = np.expand_dims(mel, axis=0)

    # Predict
    preds = model.predict(mel)
    index = np.argmax(preds)
    confidence = preds[0][index]

    return label_classes[index], confidence


# ------------------------------
# Streamlit UI
# ------------------------------
st.write("Upload a cat sound audio to classify its behavior.")

uploaded_file = st.file_uploader("Upload audio file (.wav / .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Predict"):
        with st.spinner("Analyzing sound..."):
            label, conf = predict_audio(uploaded_file)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")
