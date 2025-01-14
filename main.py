import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

model = tf.keras.models.load_model(model_path)


class_indices = json.load(open(f"{working_dir}/class_indices.json"))



def load_and_preprocess_image(image_path, target_size=(224, 224)):
 
    img = Image.open(image_path)
  
    img = img.resize(target_size)
   
    img_array = np.array(img)
   
    img_array = np.expand_dims(img_array, axis=0)
   
    img_array = img_array.astype('float32') / 255.
    return img_array



def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

remedies = {
    "Apple___Apple_scab": "Remove affected leaves and apply fungicides.",
    "Apple___Black_rot": "Prune affected branches and apply appropriate fungicides.",
    "Apple___Cedar_apple_rust": "Remove cedar trees nearby and apply fungicides.",
    "Apple___healthy": "No action required.",
    "Blueberry___healthy": "No action required.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur or potassium bicarbonate.",
    "Cherry_(including_sour)___healthy": "No action required.",
    "Corn___Cercospora_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Corn___Northern_Leaf_Blight": "Use resistant varieties and apply fungicides.",
    "Corn___healthy": "No action required.",
    "Grape___Black_rot": "Remove infected fruit and apply fungicides.",
    "Grape___Esca": "Prune out infected vines and avoid overhead irrigation.",
    "Grape___healthy": "No action required.",
    "Peach___Bacterial_spot": "Prune infected branches and apply copper-based fungicides.",
    "Peach___Powdery_mildew": "Apply sulfur or potassium bicarbonate.",
    "Peach___healthy": "No action required.",
    "Potato___Early_blight": "Rotate crops and apply fungicides.",
    "Potato___Late_blight": "Use resistant varieties and apply fungicides promptly.",
    "Potato___healthy": "No action required.",
    "Tomato___Bacterial_spot": "Remove infected plants and apply copper-based fungicides.",
    "Tomato___Late_blight": "Use resistant varieties and apply fungicides.",
    "Tomato___healthy": "No action required.",
    "Tomato___Powdery_mildew": "Apply fungicides and improve air circulation.",
    "Strawberry___Leaf_scorch": "Remove affected leaves and improve drainage.",
    "Strawberry___healthy": "No action required.",
    "Cucumber___Powdery_mildew": "Apply fungicides and ensure proper spacing.",
    "Cucumber___Downy_mildew": "Use resistant varieties and apply fungicides.",
    "Cucumber___healthy": "No action required.",
    "Bell_pepper___Bacterial_spot": "Remove infected plants and apply copper-based fungicides.",
    "Bell_pepper___Powdery_mildew": "Improve air circulation and apply fungicides.",
    "Bell_pepper___healthy": "No action required.",
    "Carrot___Cavity_spot": "Rotate crops and avoid overwatering.",
    "Carrot___healthy": "No action required.",
    "Onion___Downy_mildew": "Use resistant varieties and apply fungicides.",
    "Onion___healthy": "No action required.",
    "Soybean___Brown_spot": "Use resistant varieties and apply fungicides.",
    "Soybean___healthy": "No action required.",
    "Wheat___Leaf_rust": "Use resistant varieties and apply fungicides.",
    "Wheat___Healthy": "No action required.",
    "Rice___Bacterial_leaf_blight": "Use resistant varieties and manage water levels.",
    "Rice___Healthy": "No action required.",
    "Peas___Powdery_mildew": "Apply fungicides and improve air circulation.",
    "Peas___healthy": "No action required.",
    "Pumpkin___Powdery_mildew": "Apply fungicides and improve air circulation.",
    "Pumpkin___healthy": "No action required.",
    "Cabbage___Black_rot": "Rotate crops and apply copper-based fungicides.",
    "Cabbage___healthy": "No action required.",
    "Lettuce___Downy_mildew": "Use resistant varieties and improve air circulation.",
    "Lettuce___healthy": "No action required.",
    "Barley___Leaf_rust": "Use resistant varieties and apply fungicides.",
    "Barley___healthy": "No action required.",
    "Cotton___Bacterial_blight": "Remove infected plants and apply copper-based fungicides.",
    "Cotton___healthy": "No action required.",
    "Avocado___Root_rot": "Improve drainage and avoid overwatering.",
    "Avocado___healthy": "No action required.",
    "Citrus___Citrus_greening": "Remove infected trees and manage pests.",
    "Citrus___healthy": "No action required.",
    "Pine___Needle_cast": "Prune affected branches and improve air circulation.",
}

st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
           
            prediction = predict_image_class(model, uploaded_image, class_indices)
            remedy = remedies.get(prediction, "No remedy available.")
            st.success(f'Prediction: {str(prediction)}')
            st.write(f'Remedy: {remedy}')