##Implementation of KNN ##
## CIFAR10 dataset is used from the tensorflow##
##skleran implementation##

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.datasets import cifar10


# Load CIFAR-10 dataset from Keras
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize the image data to range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test


# Preprocess the data for kNN (flattening the images into vectors)
def preprocess_data(X_train, X_test):
    # Flatten the images (32x32x3 to 3072 dimensional vectors)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_test_flat


# Train kNN model
def train_knn(X_train_flat, y_train):
    # Initialize the k-NN classifier with k=3 (you can experiment with different values of k)
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn.fit(X_train_flat, y_train.ravel())

    return knn


# Evaluate the model
def evaluate_model(knn, X_test_flat, y_test):
    # Predict using the k-NN classifier
    y_pred = knn.predict(X_test_flat)

    # Calculate accuracy
    acc = accuracy_score(y_test.ravel(), y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test.ravel(), y_pred))


# Visualize a few test images and predictions
def visualize_results(X_test, y_test, y_pred):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.ravel()

    for i in np.arange(9):
        axes[i].imshow(X_test[i])
        axes[i].set_title(f"True: {labels[y_test.ravel()[i]]}\nPred: {labels[y_pred[i]]}")
        axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
    plt.show()


def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Preprocess data for kNN
    X_train_flat, X_test_flat = preprocess_data(X_train, X_test)

    # Train the k-NN model
    knn = train_knn(X_train_flat, y_train)

    # Evaluate the model
    evaluate_model(knn, X_test_flat, y_test)

    # Make predictions on the test set
    y_pred = knn.predict(X_test_flat)

    # Visualize some results
    visualize_results(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()
