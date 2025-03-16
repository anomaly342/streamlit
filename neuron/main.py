import numpy as np
import tensorflow as tf

# Load .npz files
train_data = np.load("neuron/k49-train-imgs.npz")
train_labels = np.load("neuron/k49-train-labels.npz")
test_data = np.load("neuron/k49-test-imgs.npz")
test_labels = np.load("neuron/k49-test-labels.npz")

# Access the arrays inside the .npz files (assuming the images are in 'arr_0' and labels in 'arr_0' as well)
train_images = train_data["arr_0"]
train_labels = train_labels["arr_0"]
test_images = test_data["arr_0"]
test_labels = test_labels["arr_0"]
print(train_images.shape)
# Check the unique label values to ensure the range is correct
print("Unique train labels:", np.unique(train_labels))  # Check the unique label values

# Ensure the images are of shape (28, 28, 1) for grayscale and normalize them to range [0, 1]
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Update the model to match the number of classes (e.g., 47 classes if max label is 46)
num_classes = len(np.unique(train_labels))  # Number of unique classes in your dataset
print(num_classes)
# Define and compile your model
model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer(
            input_shape=(28, 28, 1)
        ),  # Input layer for 28x28 grayscale images
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(
            num_classes, activation="softmax"
        ),  # Output layer adjusted to number of classes
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")
model.save("hiragana.keras")
