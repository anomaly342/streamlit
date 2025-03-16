import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tabulate import tabulate
from ydata_profiling import ProfileReport

df = pd.read_csv("machine_learning2/apple_quality.csv")

# profile = ProfileReport(df, title="Dataset Profiling Report", explorative=True)
# profile.to_file("dataset_report.html")


df = df.drop("A_id", axis=1)
df = df.dropna()

# profile = ProfileReport(df, title="Dataset Profiling Report", explorative=True)
# profile.to_file("cleansed_report.html")

x = df[
    ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]
]

y = df["Quality"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


def logistic():

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    y_pred_logreg = logreg.predict(x_test)

    print(
        "Logistic Regression Train accuracy %s" % logreg.score(x_train, y_train)
    )  # Train accuracy

    print(
        "Logistic Regression Test accuracy %s" % accuracy_score(y_pred_logreg, y_test)
    )  # Test accuracy
    print(confusion_matrix(y_test, y_pred_logreg))
    print(classification_report(y_test, y_pred_logreg))


def svc_classifier():
    # Create the SVC model
    svc = SVC(kernel="rbf")
    print("Class distribution in y_train:")
    print(y_train.value_counts())
    print("Class distribution in y_test:")
    print(y_test.value_counts())

    # Fit the model
    svc.fit(x_train, y_train)

    # Predict using the trained model
    y_pred_svc = svc.predict(x_test)
    print(y_pred_svc)
    # Print Train accuracy
    print("SVC Train accuracy: %.2f" % svc.score(x_train, y_train))

    # Print Test accuracy
    print("SVC Test accuracy: %.2f" % accuracy_score(y_test, y_pred_svc))

    # Confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))

    # Classification report
    print("Classification Report:\n", classification_report(y_test, y_pred_svc))

    pickle.dump(svc, open("apple.pkl", "wb"))


def random_forest():
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier()

    # Fit the model on the training data
    rf.fit(x_train, y_train)

    # Predict the target variable for the test data
    y_pred_rf = rf.predict(x_test)

    # Train accuracy
    print("Random Forest Train accuracy %s" % rf.score(x_train, y_train))

    # Test accuracy
    print("Random Forest Test accuracy %s" % accuracy_score(y_test, y_pred_rf))

    # Confusion matrix
    print(confusion_matrix(y_test, y_pred_rf))

    # Classification report (precision, recall, F1-score, etc.)
    print(classification_report(y_test, y_pred_rf))


def knn():
    # Initialize the KNN classifier (you can specify the number of neighbors, e.g., n_neighbors=5)
    knn = KNeighborsClassifier(
        n_neighbors=12
    )  # You can tune 'n_neighbors' for better performance

    # Fit the model to the training data
    knn.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred_knn = knn.predict(x_test)

    # Print the training accuracy
    print("KNN Train accuracy %s" % knn.score(x_train, y_train))  # Train accuracy

    # Print the test accuracy
    print("KNN Test accuracy %s" % accuracy_score(y_test, y_pred_knn))  # Test accuracy

    # Print confusion matrix
    print(confusion_matrix(y_test, y_pred_knn))

    # Print classification report (precision, recall, f1-score)
    print(classification_report(y_test, y_pred_knn))


def naive_bayes():
    # Initialize the Naive Bayes classifier (GaussianNB)
    nb = GaussianNB()

    # Fit the model on the training data
    nb.fit(x_train, y_train)

    # Predict the target variable for the test data
    y_pred_nb = nb.predict(x_test)

    # Train accuracy
    print("Naive Bayes Train accuracy %s" % nb.score(x_train, y_train))

    # Test accuracy
    print("Naive Bayes Test accuracy %s" % accuracy_score(y_test, y_pred_nb))

    # Confusion matrix
    print(confusion_matrix(y_test, y_pred_nb))

    # Classification report (precision, recall, F1-score, etc.)
    print(classification_report(y_test, y_pred_nb))


logistic()
svc_classifier()
knn()
naive_bayes()
random_forest()
