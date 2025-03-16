import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Title and description
st.title("Model Comparison and Dataset Training Overview")
st.markdown(
    """
This page provides a comparison between various classification models based on their performance using the apple quality dataset. I choosed the **Support Vector Classifier (SVC)** as the best model because of its high accuracy and balanced performance across different metrics.
"""
)

# Displaying the model comparison in a table
st.subheader("Model Comparison")

# Table for Model Performance
model_comparison_data = {
    "Model": ["SVC", "KNN", "Naive Bayes", "Random Forest", "Logistic Regression"],
    "Train Accuracy": [0.90, 0.91, 0.75, 1.0, 0.74],
    "Test Accuracy": [0.90, 0.90, 0.75, 0.90, 0.75],
    "Precision (Bad)": [0.89, 0.88, 0.74, 0.90, 0.75],
    "Recall (Bad)": [0.90, 0.92, 0.76, 0.89, 0.75],
    "Precision (Good)": [0.90, 0.92, 0.76, 0.90, 0.75],
    "Recall (Good)": [0.89, 0.87, 0.73, 0.90, 0.76],
    "Overall Accuracy": [0.90, 0.90, 0.75, 0.90, 0.75],
}

# Creating a DataFrame to display in a table format
model_comparison_df = pd.DataFrame(model_comparison_data)
st.write(model_comparison_df)

# Confusion Matrix for SVC
st.subheader("Confusion Matrix for SVC")
# Example confusion matrix data for SVC
conf_matrix = np.array([[533, 60], [65, 542]])
# Plotting the confusion matrix
fig, ax = plt.subplots()
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Bad", "Good"],
    yticklabels=["Bad", "Good"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - SVC")
st.pyplot(fig)

# Model names and accuracy values for graph
models = ["SVC", "KNN", "Naive Bayes", "Random Forest", "Logistic Regression"]
train_accuracy = [0.90, 0.91, 0.75, 1.0, 0.74]
test_accuracy = [0.90, 0.90, 0.75, 0.90, 0.75]

# Interactive bar chart for model comparison
fig = go.Figure(
    data=[
        go.Bar(
            x=models,
            y=train_accuracy,
            name="Train Accuracy",
            hovertext=[f"Train Accuracy: {acc:.2f}" for acc in train_accuracy],
            hoverinfo="text",  # Display hover text
            marker=dict(color="white"),
        ),
        go.Bar(
            x=models,
            y=test_accuracy,
            name="Test Accuracy",
            hovertext=[f"Test Accuracy: {acc:.2f}" for acc in test_accuracy],
            hoverinfo="text",  # Display hover text
            marker=dict(color="green"),
        ),
    ]
)

# Customize layout
fig.update_layout(
    title="Train vs Test Accuracy Comparison",
    xaxis_title="Models",
    yaxis_title="Accuracy",
    barmode="group",  # Group the bars
    hovermode="closest",  # Display hover info closest to cursor
)

# Display interactive chart in Streamlit
st.plotly_chart(fig)


# Providing a detailed explanation for choosing SVC
st.subheader("Why Support Vector Classifier (SVC) was chosen?")
st.markdown(
    """
- The **Support Vector Classifier (SVC)** is a robust machine learning model used for classification tasks. It works by finding the optimal hyperplane that maximizes the margin between classes in a high-dimensional space.
- **SVC** provides the best balance of accuracy, precision, and recall, showing consistent performance in both training and test sets.
- The model achieves **90% accuracy** on both the training and test sets, so it has high generalization.
- Precision and recall for both classes (bad and good) are strong, making it a reliable classifier for this dataset.
"""
)

# Including details about the dataset used
st.subheader("Dataset Overview")
st.markdown(
    """
The dataset used in this analysis is from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality), which contains information about the quality of apples based on different features such as color, texture, size, and weight. The dataset provides labels categorizing apples as either **bad** or **good** quality.

The dataset was pre-processed and split into training and test sets. Features were scaled and cleaned to ensure the models could learn effectively, with the aim of classifying apples into these two categories based on the given attributes.
"""
)

# Basic Theory of Support Vector Classifier (SVC)
st.subheader("Support Vector Classifier (SVC)")
st.markdown(
    """
The **Support Vector Classifier (SVC)** is a type of supervised learning algorithm used for classification tasks. It belongs to the family of Support Vector Machines (SVMs), which work by finding the optimal hyperplane that maximizes the margin between classes. In simple terms, the SVC aims to find a decision boundary that separates data points of different classes in a way that minimizes classification errors.

Key points about SVC:
- **Hyperplane**: The decision boundary that separates different classes in the feature space.
- **Margin**: The distance between the closest points of the classes and the hyperplane. SVC maximizes this margin to ensure better generalization.
- **Kernel trick**: SVC can use different kernel functions (linear, polynomial, radial basis function) to map data to higher dimensions, making it effective in non-linear classification tasks.
- **C parameter**: Controls the trade-off between maximizing the margin and minimizing classification errors.
- **Gamma parameter**: Determines the influence of each data point on the decision boundary.

The SVC is particularly effective for high-dimensional datasets, and its ability to handle non-linear boundaries makes it a powerful tool in classification problems such as this one.

[Source](https://www.geeksforgeeks.org/support-vector-machine-algorithm/)
"""
)

# Explanation about data preprocessing
st.subheader("Data Preprocessing")
st.markdown(
    """
Before feeding the data into machine learning models, I performed some basic data preprocessing steps. One of these steps involved dropping the **'A_id'** column, which serves as an identification field for each apple in the dataset. Since this column does not contain any useful information about the quality of the apples, it was removed during the data cleansing process.
"""
)

# Code Snippet for Data Preprocessing
st.subheader("Data Preprocessing Code")
st.code(
    """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("machine_learning2/apple_quality.csv")

# Drop the 'A_id' column as it is just an identifier
df = df.drop("A_id", axis=1)

# Drop missing values
df = df.dropna()

# Define features and target variable
x = df[["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]]
y = df["Quality"]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
"""
)
