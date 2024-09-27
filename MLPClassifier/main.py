from MLPClassifier.core.mlp_classifier import MLP, AutoencoderMLP
from MNIST_utils.data_loader import load_mnist
from MNIST_utils.preprocessing import encode_labels, split_data
from MNIST_utils.visualization import plot_confusion_matrix
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
    print(f"\nTotal Accuracy: {accuracy * 100:.2f}%\n")
    print("Total Error: ", error_rate)

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

    num_classes = Y.shape[1]

    # MLP Configuration
    # hidden_layer_sizes = (710,)  # PCA Technique result
    
    use_autoencoder = False
    input_size = X_train.shape[1]
    autoencoder_hidden_size = 200
    autoencoder_epochs = 20
    learning_rate = 0.95
    num_epochs = 1000
    epochs_step = 100
    loss = 'mse'
    shuffle_data = True
    early_stopping = None
    momentum = 0.9
    heuristic = None

    # Uncomment the heuristic you want to use:
    # heuristic = {'type': 'mean'}
    # heuristic = {'type': 'sqrt'}
    # heuristic = {'type': 'kolmogorov'}
    # heuristic = {'type': 'fletcher_gloss'}
    # heuristic = {'type': 'baum_haussler'}
    # heuristic = {'type': 'exhaustive', 'step': 100}
    # heuristic = {'type': 'pca', 'variance_ratio': 0.95}

    # Create the MLP Classifier
    if use_autoencoder:
        print("Using Autoencoder + MLP")
        classifier = AutoencoderMLP(input_size, autoencoder_hidden_size, (100,),
                                    learning_rate=learning_rate, autoencoder_epochs=autoencoder_epochs,
                                    mlp_epochs=1000, mlp_epochs_step=1000,
                                    loss=loss, shuffle_data=shuffle_data,
                                    early_stopping=early_stopping, momentum=momentum,
                                    heuristic=heuristic)
        classifier_type = "autoencoder_mlp"
    else:
        print("Using MLP without Autoencoder")
        classifier = MLP(hidden_layer_sizes=(heuristic,), learning_rate=learning_rate,
                         num_epochs=1000, epochs_step=100, loss=loss,
                         shuffle_data=shuffle_data, early_stopping=early_stopping, momentum=momentum,
                         heuristic={'type': 'mean'})
        classifier_type = "mlp"

    # Train the classifier
    train_classifier(classifier, X_train, Y_train, X_val=X_test, y_val=Y_test, plot=True)

    # Make predictions on the test data
    predictions = classifier.predict(X_test)

    # Convert one-hot encoded predictions back to class labels
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)
    y_test = np.argmax(Y_test, axis=1)

    # Evaluate the classifier performance
    evaluate_classifier(predictions, y_test, num_classes, classifier_type)

if __name__ == "__main__":
    main()
