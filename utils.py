import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

def load_model_and_scaler():
    model = tf.keras.models.load_model("heart_disease_model.h5")
    scaler = joblib.load("scaler.pkl")
    return scaler, model

def preprocess_input_data(data, scaler):
    # Ensure column order matches training
    numerical_columns_idx = [0, 3, 4, 7, 9]  # Indices of the 5 numerical columns
    scaled_data = data.copy()

    # Scale only numerical columns
    scaled_data[:, numerical_columns_idx] = scaler.transform(data[:, numerical_columns_idx])
    return scaled_data


def generate_ecg_images(patient_data):
    ecg_images = np.zeros((1, 64, 64, 3))
    time_points = np.linspace(0, 2*np.pi, 64)

    hr_pattern = np.sin(time_points * patient_data[0][7]/100) * 0.5
    bp_pattern = np.sin(time_points * patient_data[0][3]/50) * 0.3
    chol_pattern = np.sin(time_points * patient_data[0][4]/200) * 0.6
    age_pattern = np.sin(time_points * patient_data[0][0]/50) * 0.2
    ex_pattern = np.sin(time_points * (patient_data[0][8]+1)*2) * 0.4
    st_pattern = np.sin(time_points * patient_data[0][9]*2) * 0.4

    for i in range(64):
        ecg_images[0, i, :, 0] = (hr_pattern[i] + bp_pattern[i] + 1) * 127
        ecg_images[0, i, :, 1] = (chol_pattern[i] + age_pattern[i] + 1) * 127
        ecg_images[0, i, :, 2] = (ex_pattern[i] + st_pattern[i] + 1) * 127

    return ecg_images / 255.0

def create_patient_sequences(data, sequence_length=5):
    sequences = np.zeros((1, sequence_length, data.shape[1]))
    for t in range(sequence_length):
        sequences[0, t, :] = data[0] * (0.9 + 0.1 * np.random.rand())
    return sequences
