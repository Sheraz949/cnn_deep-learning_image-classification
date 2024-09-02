import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('food_classification_model.h5')

# Function to preprocess and predict
def predict(image):
    img = image.resize((180, 180))
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    return predictions

# Streamlit app
st.title("Food Image Classification")
st.write("Upload an image of food and the model will predict the class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    predictions = predict(image)
    class_names = ["apple_pie", "cheesecake", "chicken_curry", "french_fries", "fried_rice", "hamburger", "hot_dog", "ice_cream", "omelette", "pizza", "sushi"]  # Replace with actual class names
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")
