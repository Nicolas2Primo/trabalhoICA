import numpy as np
import matplotlib.pyplot as plt


class MLPRegressor(object):
    """
    A Multi-Layer Perceptron (MLP) Regressor for regression tasks.

    Parameters:
        hidden_layer_sizes (tuple or list): The number of neurons in each hidden layer.
                                           Default is (100,).
        learning_rate (float): The learning rate used in gradient descent during training.
                               Default is 0.01.
        num_epochs (int): The number of training epochs. Each epoch represents a complete
                          iteration over the entire dataset during training. Default is 1000.
        epochs_step (int): The interval at which training progress is displayed. Default is 10.
        shuffle_data (bool): Whether to shuffle the training data before each epoch.
                             Default is True.
        early_stopping (dict): Dictionary with keys 'patience' and 'delta'. If provided,
                               early stopping will be used to prevent overfitting.
                               'patience': Number of epochs to wait for improvement.
                               'delta': Minimum change in loss to be considered an improvement.

    Attributes:
        hidden_layer_sizes (tuple): The number of neurons in each hidden layer.
        learning_rate (float): The learning rate used in gradient descent during training.
        num_epochs (int): The number of training epochs.
        epochs_step (int): The interval at which training progress is displayed.
        weights (list): A list of weight matrices for each layer in the network.
        biases (list): A list of bias vectors for each layer in the network.
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, num_epochs=1000,
                 epochs_step=10, shuffle_data=True, early_stopping=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.epochs_step = epochs_step
        self.shuffle_data = shuffle_data
        self.early_stopping = early_stopping
        self.weights = None
        self.biases = None

    def linear_activation(self, x):
        """
        Apply the identity (linear) activation function element-wise to the input array.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: The array with the identity activation applied element-wise.
        """
        return x

    def initialize_parameters(self, num_features):
        """
        Initialize the weights and biases for the neural network.

        Parameters:
            num_features (int): The number of input features.

        Returns:
            None
        """
        layer_sizes = [num_features] + list(self.hidden_layer_sizes) + [1]  # Regression has a single output neuron
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def sigmoid(self, z):
        # Clip the input to prevent overflow or underflow issues
        # z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        """
        Perform forward propagation for the neural network.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                            containing the input samples.

        Returns:
            list: A list containing the activations of each layer in the network.
        """
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self.sigmoid(z))

        # For the output layer, use linear activation for regression.
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        activations.append(self.linear_activation(z_output))

        return activations


    def compute_loss(self, y_true, y_pred):
        """
        Compute the mean squared error (MSE) loss between true labels and predicted values.

        Parameters:
            y_true (numpy.ndarray): The true labels of shape (num_samples,).
            y_pred (numpy.ndarray): The predicted values of shape (num_samples,).

        Returns:
            float: The computed MSE loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward_propagation(self, X, y, activations):
        """
        Perform backward propagation for the neural network to compute gradients.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                               containing the input samples.
            y (numpy.ndarray): The target vector of shape (num_samples,) containing the regression values.
            activations (list): A list containing the activations of each layer in the network.

        Returns:
            tuple: A tuple containing the gradients of weights and biases for each layer.
        """
        num_samples = X.shape[0]
        num_layers = len(self.weights)
        deltas = [activations[-1] - y.reshape(-1, 1)]  # Ensure deltas[-1] has shape (num_samples, 1)

        dW = [None] * num_layers
        dB = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            dW[i] = np.dot(activations[i].T, deltas[-1]) / num_samples
            dB[i] = np.sum(deltas[-1], axis=0) / num_samples

            if i > 0:
                deltas.append(np.dot(deltas[-1], self.weights[i].T) * (activations[i] * (1 - activations[i])))

        return dW, dB


    def train(self, X, y, plot=False):
        """
        Train the MLP regressor using the provided training data.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                            containing the training samples.
            y (numpy.ndarray): The target vector of shape (num_samples,) containing the regression values.
            plot (boolean): Option to display the convergence (loss) curve at the end of the training. Default is False.

        Returns:
            None
        """
        num_samples, num_features = X.shape

        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)  # Convert to tuple if the user provides an integer

        if self.early_stopping is not None:
            patience = self.early_stopping.get('patience', 10)
            delta = self.early_stopping.get('delta', 1e-4)
            best_loss = np.inf
            no_improvement_count = 0

        self.initialize_parameters(num_features)

        loss_values = []  # List to store loss values at each epoch

        for epoch in range(1, self.num_epochs + 1):
            if self.shuffle_data:
                # Shuffle the training data before each epoch.
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

            activations = self.forward_propagation(X)
            dW, dB = self.backward_propagation(X, y, activations)

            # Update weights and biases using gradient descent.
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * dB[i]

            # Compute and display the loss at the specified intervals.
            loss = self.compute_loss(y, activations[-1])

            # Check for early stopping.
            if self.early_stopping is not None:
                if loss + delta < best_loss:
                    best_loss = loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Early stopping: No improvement in the last {0} epochs.".format(patience))
                        break

            if epoch % self.epochs_step == 0 or epoch == self.num_epochs:
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {loss:.5f}")
                loss_values.append(loss)

        if plot:
            # Plot convergence curve (loss) during training
            plt.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.show()

    def predict(self, X):
        """
        Predict the regression values for the given input samples.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                               containing the samples for prediction.

        Returns:
            numpy.ndarray: The predicted regression values as an array of shape (num_samples,)
        """
        activations = self.forward_propagation(X)
        return activations[-1].flatten()  # Return the output as a 1D array
