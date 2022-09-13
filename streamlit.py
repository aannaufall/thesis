import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as mobilenet_v2_preprocess_input



st.header("Indentifikasi Varietas Mangga Berdasarkan Citra Daun Menggunakan CNN ")
model = tf.keras.models.load_model("/content/gdrive/MyDrive/ColabNotebooks/model/ModelVGG16epoch10.h5")

### load file
uploaded_file = st.file_uploader("Pilih file gambar", type=['jpg', 'png', 'jpeg'])

map_dict = {0: 'Mangga Apel',
            1: 'Mangga Gedong',
            2: 'Mangga Golek',
            3: 'Mangga Lalijiwo',
            4: 'Mangga Manalagi',
            5: 'Mangga Wirasangka',}
            

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    st.image(opencv_image, channels="RGB")
    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Klik Untuk Identifikasi")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("Varietas {}".format(map_dict [prediction]))