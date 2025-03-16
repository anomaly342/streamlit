import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Title and description
st.title("Hiragana Character Recognition with CNN")
st.markdown(
    """
    This page provides an overview of a **Convolutional Neural Network (CNN)** trained on the **Kuzushiji-49 (K49)** dataset for recognizing handwritten Hiragana characters.
    The dataset is obtained from [KMNIST GitHub Repository](https://github.com/rois-codh/kmnist).
    """
)

# Code Snippet
st.subheader("Model Training Code")
st.code(
    """
import numpy as np
import tensorflow as tf

# Load .npz files
data_path = "neuron/k49-"
train_images = np.load(f"{data_path}train-imgs.npz")['arr_0']
train_labels = np.load(f"{data_path}train-labels.npz")['arr_0']
test_images = np.load(f"{data_path}test-imgs.npz")['arr_0']
test_labels = np.load(f"{data_path}test-labels.npz")['arr_0']

# Normalize images
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

# Define the CNN model
num_classes = len(np.unique(train_labels))
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)

# Save the model
model.save("hiragana.keras")
    """,
    language="python",
)

# CNN Theory
st.subheader("Convolutional Neural Networks (CNNs)")
st.markdown(
    """
    CNNs are specialized deep learning models designed for image processing. They consist of:
    
    - **Convolutional Layers**: Extract features using filters.
    - **Pooling Layers**: Reduce spatial dimensions while retaining important features.
    - **Fully Connected Layers**: Make final predictions based on extracted features.
    
    CNNs excel at recognizing patterns in images.

    [Source](https://www.geeksforgeeks.org/introduction-convolution-neural-network/)
    """
)

# Conclusion
st.subheader("Conclusion")
st.markdown(
    """
    This CNN model successfully recognizes handwritten Hiragana characters using the **K49 dataset**. 
    The trained model achieves high accuracy and can be extended for real-world applications in Japanese OCR systems.
    """
)
