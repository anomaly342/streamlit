import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform  # <-- Use this for OS detection

# Clear session to avoid conflicts
tf.keras.backend.clear_session()

# Load the trained model
model = tf.keras.models.load_model("models/hiragana.keras")

# Load class map for hiragana characters
hiragana_df = pd.read_csv("models/k49_classmap.csv")

# Detect the OS and set Japanese font
os_name = platform.system()

if os_name == "Windows":
    font_path = "C:/Windows/Fonts/msgothic.ttc"  # Windows (MS Gothic)
elif os_name == "Darwin":  # macOS
    font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # macOS default
else:  # Linux
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Noto Sans

# Load the font properties
jp_font = fm.FontProperties(fname=font_path)


# Function to preprocess the image and predict
def predict_image(image):
    # Convert image to grayscale
    image = image.convert("L")

    # Convert to numpy array and normalize
    image_array = np.array(image).astype("float32") / 255.0

    # Reshape for model input (28, 28, 1)
    image_array = image_array.reshape(-1, 28, 28, 1)

    # Predict class
    predictions = model.predict(image_array)[0]  # Get first prediction
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions


# Function to get hiragana character from class map
def get_hiragana_char(predicted_class):
    return hiragana_df.iloc[predicted_class]["char"]


# Streamlit App for Hiragana Recognition
st.title("Hiragana Character Recognition")

st.write("Upload a **28x28** image of a Hiragana character and click 'Predict'.")

# File uploader for the image
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_image)

    # Check if the image is 28x28 pixels
    if img.size != (28, 28):
        st.error(
            f"❌ The uploaded image size is {img.size}. Please upload an image that is **exactly 28x28 pixels.**"
        )
    else:
        # Display the uploaded image
        st.image(img, caption="✅ Uploaded Image (28x28)", use_column_width=True)

        # Button to predict the drawn character
        if st.button("Predict"):
            # Predict the character
            predicted_class, predictions = predict_image(img)

            # Get the Hiragana character corresponding to the predicted class
            hiragana_char = get_hiragana_char(predicted_class)

            # Show the predicted result
            st.subheader(f"Predicted Hiragana character: {hiragana_char}")

            # Plot and show probabilities
