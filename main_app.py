import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import cv2

# Loading the Model
model = load_model('/home/abdelmoneim/التنزيلات/model_CNN.h5')

# Name of Classes
CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
 'Potato___Early_blight', 'Potato___Late_blight' 'Potato___healthy',
 'Tomato_Bacterial_spot' ,'Tomato_Early_blight' ,'Tomato_Late_blight',
 'Tomato_Leaf_Mold' ,'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus' ,'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

# Setting Title of App
st.title("🌿 Plant Disease Detection")
st.markdown("Upload an image of the plant leaf to detect the disease")

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# On predict button click
if st.button('Predict'):

    if plant_image is not None:
        with st.spinner('Processing...'):
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR", caption='Uploaded Image', use_column_width=True)

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (256, 256))
            # Convert image to 4 Dimension
            opencv_image = np.expand_dims(opencv_image, axis=0)

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]

            # Display the result
            st.success(f'**Prediction:** {result}')

    else:
        st.error("Please upload an image file")
