import streamlit as st
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model (apple.pkl)
with open("models/apple.pkl", "rb") as file:
    model = pickle.load(file)

# Create a LabelEncoder for the "Quality" column if necessary
label_encoder = LabelEncoder()
label_encoder.classes_ = ["bad", "good"]  # Ensure the encoder knows both labels


def predict_quality(features):
    # Ensure the features are a numpy array and reshape to 2D (1 sample, n features)
    features_array = np.array(features).reshape(1, -1)  # reshape to (1, n_features)

    prediction = model.predict(features_array)
    return prediction[0]


# Streamlit app layout
st.title("Apple Quality Prediction App")

# Sidebar: Add buttons to simulate tabs


st.write("Enter the features of the apple to predict its quality.")

# Input fields for each feature using sliders
size = st.slider("Size", min_value=-5.0, max_value=5.0, step=0.01)
weight = st.slider("Weight", min_value=-5.0, max_value=5.0, step=0.01)
sweetness = st.slider("Sweetness", min_value=-5.0, max_value=5.0, step=0.01)
crunchiness = st.slider("Crunchiness", min_value=-5.0, max_value=5.0, step=0.01)
juiciness = st.slider("Juiciness", min_value=0.0, max_value=5.0, step=0.01)
ripeness = st.slider("Ripeness", min_value=-5.0, max_value=5.0, step=0.01)
acidity = st.slider("Acidity", min_value=-5.0, max_value=5.0, step=0.01)

# Button to make prediction
if st.button("Predict Quality"):
    # Prepare the input data for prediction
    input_data = [size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]
    # Get the prediction
    quality = predict_quality(input_data)
    # Display the result
    st.write(f"The predicted quality of the apple is: {quality}")
