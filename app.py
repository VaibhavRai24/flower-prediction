import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import streamlit as st

import numpy as np
st.header('Flower Classification using the CNN')

flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model(r"C:\Users\VAIBHAVRAI\OneDrive\Desktop\ml projects\Flower_recognition.h5")

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dinm = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dinm)
    results = tf.nn.softmax(predictions[0])

    outcome = 'The image belongs to ' + flower_names[np.argmax(results)] + ' with a score of ' + str(
        np.max(results) * 100)
    return outcome

uploaded_file  = st.file_uploader("Upload an Image")
if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())