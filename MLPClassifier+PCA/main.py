from MLPClassifier.core.mlp_classifier import MLP
from MNIST_utils.data_loader import load_mnist
from MNIST_utils.preprocessing import encode_labels, split_data
from MNIST_utils.visualization import plot_confusion_matrix
from MNIST_utils import pca
import numpy as np
import matplotlib.pyplot as plt
import time

def train_classifier(classifier, X, y, X_val=None, y_val=None, plot=False):
    start_time = time.time()
    classifier.train(X, y, X_val, y_val, plot)
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

def evaluate_classifier(predictions, test_labels, num_classes):
    # Calculate accuracy
    accuracy = np.mean(predictions == test_labels)
    error_rate = 1 - accuracy
    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Error Rate: {error_rate * 100:.2f}%")

    # Calculate accuracy for each class
    for i in range(num_classes):
        class_mask = test_labels == i
        class_accuracy = np.mean(predictions[class_mask] == test_labels[class_mask])
        print(f"Class {i}: Accuracy: {class_accuracy * 100:.2f}%")

    # Compute and plot the confusion matrix
    plot_confusion_matrix(test_labels, predictions, 
                          labels=range(num_classes), 
                          save_path='MLPClassifier/results/mlp_confusion_matrix.png')

def main():
    # Load and prepare the MNIST data
    X, y = load_mnist(normalize=True)
    Y, lb = encode_labels(y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Apply PCA
    variance_ratio = 0.95  # Keep 95% of the variance
    X_train_pca, X_test_pca, pca_components = pca.apply_pca(X_train, X_test, variance_ratio)

    num_classes = Y.shape[1]

    # MLP Configuration
    learning_rate = 0.95  # Reduced from 0.95
    num_epochs = 1000
    epochs_step = 100
    loss = 'mse'  # Changed from 'mse' for multi-class classification
    shuffle_data = True
    early_stopping = None  # Added early stopping
    momentum = 0.9

    # Create the MLP Classifier
    classifier = MLP(hidden_layer_sizes=('pca',),  # Use number of PCA components
                     learning_rate=learning_rate,
                     num_epochs=num_epochs,
                     epochs_step=epochs_step,
                     loss=loss,
                     shuffle_data=shuffle_data,
                     early_stopping=early_stopping,
                     momentum=momentum)

    # Train the classifier
    train_classifier(classifier, X_train_pca, Y_train, X_val=X_test_pca, y_val=Y_test, plot=True)

    # Make predictions on the test data
    predictions = classifier.predict(X_test_pca)

    # Convert one-hot encoded predictions back to class labels
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    y_test = np.argmax(Y_test, axis=1)

    # Evaluate the classifier performance
    evaluate_classifier(predictions, y_test, num_classes)

if __name__ == "__main__":
    main()
