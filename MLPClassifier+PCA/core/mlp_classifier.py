import numpy as np
import matplotlib.pyplot as plt


class MLP(object):
    """
    A Multi-Layer Perceptron (MLP) Classifier for binary and multiclass classification tasks.

    Parameters:
        hidden_layer_sizes (tuple or list): The number of neurons in each hidden layer.
                                           Default is (100,).
        learning_rate (float): The learning rate used in gradient descent during training.
                               Default is 0.01.
        num_epochs (int): The number of training epochs. Each epoch represents a complete
                          iteration over the entire dataset during training. Default is 1000.
        epochs_step (int): The interval at which training progress is displayed. Default is 10.
        loss (str): The loss function to use. Options: 'cross-entropy' or 'mse' (default).
        shuffle_data (bool): Whether to shuffle the training data before each epoch.
                             Default is True.
        early_stopping (dict): Dictionary with keys 'patience' and 'delta'. If provided,
                               early stopping will be used to prevent overfitting.
                               'patience': Number of epochs to wait for improvement.
                               'delta': Minimum change in loss to be considered an improvement.
        heuristic (dict): Dictionary with keys 'type' and 'step'. If provided, it configures
                          the heuristic search for hidden layer sizes.
                          'type': The type of heuristic to use for finding hidden layer sizes.
                                  Options: 'all', 'pca', 'exhaustive', or None (default).
                                  'all': Uses heuristic rules to find the best number of hidden neurons.
                                  'pca': Uses PCA-based technique to estimate the number of hidden neurons.
                                  'exhaustive': Performs an exhaustive search for the best number of hidden neurons.
                          'step': The step size for the exhaustive search. Applicable only when 'type' is 'exhaustive'.
                          'variance_ratio': (float, optional): The maximum cumulative variance ratio to consider
                                              during PCA. Defaults to 0.95, which keeps 95% of the variance.

    Attributes:
        hidden_layer_sizes (tuple): The number of neurons in each hidden layer.
        learning_rate (float): The learning rate used in gradient descent during training.
        num_epochs (int): The number of training epochs.
        epochs_step (int): The interval at which training progress is displayed.
        loss (str): The loss function used for training.
        weights (list): A list of weight matrices for each layer in the network.
        biases (list): A list of bias vectors for each layer in the network.
        heuristic (dict): The configuration for the heuristic search of hidden layer sizes.
    """

    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, num_epochs=1000,
                 epochs_step=10, loss='mse', shuffle_data=True, early_stopping=None, momentum=0.9,
                 heuristic=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.epochs_step = epochs_step
        self.loss = loss
        self.shuffle_data = shuffle_data
        self.early_stopping = early_stopping
        self.weights = None
        self.biases = None
        self.momentum = momentum
        
        if heuristic is None:
            heuristic = {'type': 'all', 'step': 100, 'variance_ratio': 0.95}
        self.heuristic = heuristic

    def sigmoid(self, x):
        """
        Apply the sigmoid function element-wise to the input array.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: The array with sigmoid function applied element-wise.
        """
        return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

    def initialize_parameters(self, num_features, num_classes):
        """
        Initialize the weights and biases for the neural network.

        Parameters:
            num_features (int): The number of input features.
            num_classes (int): The number of output classes.

        Returns:
            None
        """
        layer_sizes = [num_features] + list(self.hidden_layer_sizes) + [num_classes]
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        self.biases = [np.zeros(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

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

        # For the output layer, use softmax for cross-entropy loss and sigmoid for mse loss.
        output_layer_activation = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        if self.loss == 'cross-entropy':
            activations.append(self.softmax(output_layer_activation))
        elif self.loss == 'mse':
            activations.append(self.sigmoid(output_layer_activation))
        else:
            raise ValueError(f"Invalid loss function: {self.loss}. Use 'cross-entropy' or 'mse'.")

        return activations

    def compute_loss(self, y_true, y_pred):
        """
        Compute the loss between true labels and predicted probabilities.

        Parameters:
            y_true (numpy.ndarray): The true labels of shape (num_samples, num_classes).
            y_pred (numpy.ndarray): The predicted probabilities of shape (num_samples, num_classes).

        Returns:
            float: The computed loss value.
        """
        if self.loss == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.loss == 'cross-entropy':
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Avoid log(0) issues
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError(f"Invalid loss function: {self.loss}. Use 'cross-entropy' or 'mse'.")

    def backward_propagation(self, X, y, activations):
        """
        Perform backward propagation for the neural network to compute gradients.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                               containing the input samples.
            y (numpy.ndarray): The target matrix of shape (num_samples, num_classes)
                               containing the class labels.
            activations (list): A list containing the activations of each layer in the network.

        Returns:
            tuple: A tuple containing the gradients of weights and biases for each layer.
        """
        num_samples = X.shape[0]
        num_layers = len(self.weights)
        deltas = [activations[-1] - y]

        dW = [None] * num_layers
        dB = [None] * num_layers

        for i in range(num_layers - 1, -1, -1):
            dW[i] = np.dot(activations[i].T, deltas[-1]) / num_samples
            dB[i] = np.sum(deltas[-1], axis=0) / num_samples

            if i > 0:
                deltas.append(np.dot(deltas[-1], self.weights[i].T) * (activations[i] * (1 - activations[i])))

        return dW, dB
    

    def train(self, X, y, X_val=None, y_val=None, plot=False):
        """
        Train the MLP classifier using the provided training data.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                            containing the training samples.
            y (numpy.ndarray): The target matrix of shape (num_samples, num_classes)
                            containing the class labels.
            X_val (numpy.ndarray): The feature matrix of shape (num_val_samples, num_features)
                                containing the validation samples (optional).
            y_val (numpy.ndarray): The target matrix of shape (num_val_samples, num_classes)
                                containing the validation class labels (optional).
            plot (boolean): Option to display the convergence (loss) and accuracy curve
                         at the end of the training. Default is False.

        Returns:
            None
        """
        num_samples, num_features = X.shape
        num_classes = y.shape[1]

        if isinstance(self.hidden_layer_sizes, int):
            self.hidden_layer_sizes = (self.hidden_layer_sizes,)  # Convert to tuple if the user provides an integer

        if self.early_stopping is not None:
            patience = self.early_stopping.get('patience', 10)
            delta = self.early_stopping.get('delta', 1e-4)
            best_loss = np.inf
            no_improvement_count = 0

        # Search for the best number of hidden neurons if not specified
        if len(self.hidden_layer_sizes) == 1 and isinstance(self.hidden_layer_sizes[0], str):
            if self.hidden_layer_sizes[0] == 'heuristic':
                print("Using heuristic rules to find the best number of hidden neurons.")
                self.hidden_layer_sizes = self._find_best_hidden_neurons(X, y, X_val, y_val, heuristic_type=self.heuristic.get('type'))
            elif self.hidden_layer_sizes[0] == 'pca':
                print("Using PCA-based technique to estimate the number of hidden neurons.")
                self.hidden_layer_sizes = self._find_hidden_neurons_pca(X, y, variance_ratio=self.heuristic.get('variance_ratio'))
            elif self.hidden_layer_sizes[0] == 'exhaustive':
                print("Performing an exhaustive search for the best number of hidden neurons.")
                self.hidden_layer_sizes = self._exhaustive_search(X, y, X_val, y_val, step=self.heuristic.get('step'))

        self.initialize_parameters(num_features, num_classes)

        # Initialize momentum terms for weights and biases to zeros
        v_dW = [np.zeros_like(weight) for weight in self.weights]
        v_dB = [np.zeros_like(bias) for bias in self.biases]

        loss_values = []  # List to store loss values at each epoch
        accuracies = []   # List to store accuracies at each epoch

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
                # self.weights[i] -= self.learning_rate * dW[i]
                # self.biases[i] -= self.learning_rate * dB[i]
                v_dW[i] = self.learning_rate * dW[i] + self.momentum * v_dW[i]
                v_dB[i] = self.learning_rate * dB[i] + self.momentum * v_dB[i]
                self.weights[i] -= v_dW[i]
                self.biases[i] -= v_dB[i]

            # Compute and display the loss at the specified intervals.
            loss = self.compute_loss(y, activations[-1])

            # Calculate accuracy on the validation set
            current_accuracy = 0
            if X_val is not None and y_val is not None:
                activations_val = self.forward_propagation(X_val)
                predictions_val = self.predict(X_val)
                accuracy_val = np.mean(predictions_val == np.argmax(y_val, axis=1))
                current_accuracy = accuracy_val

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
                print(f"Epoch {epoch}/{self.num_epochs} - Loss: {loss:.5f} - Accuracy: {current_accuracy:.5f}")
                # Append the accuracy and loss value for this epoch
                accuracies.append(current_accuracy)
                loss_values.append(loss)
        
        if plot:
            # Plot convergence curve (loss) and accuracy during training
            plt.plot(range(1, len(loss_values) + 1), loss_values, color='blue', label='Loss')
            if X_val is not None and y_val is not None:
                plt.plot(range(1, len(accuracies) + 1), accuracies, color='green', label='Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.show()
            
    def predict(self, X):
        """
        Predict the class labels for the given input samples.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                               containing the samples for prediction.

        Returns:
            numpy.ndarray: The predicted class labels as an array of shape (num_samples,)
        """
        activations = self.forward_propagation(X)
        probabilities = activations[-1]

        # For binary classification, return the class with the highest probability.
        if probabilities.shape[1] == 2:
            return (probabilities[:, 1] > 0.5).astype(int)
        # For multiclass classification, return the class with the highest probability for each sample.
        else:
            return np.argmax(probabilities, axis=1)

    def _find_best_hidden_neurons(self, X, y, X_val=None, y_val=None, heuristic_type='all'):
        """
        Use heuristic rules for preliminary tests and select the best number of hidden neurons.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                            containing the training samples.
            y (numpy.ndarray): The target matrix of shape (num_samples, num_classes)
                            containing the class labels.
            X_val (numpy.ndarray): The feature matrix of shape (num_val_samples, num_features)
                                containing the validation samples (optional).
            y_val (numpy.ndarray): The target matrix of shape (num_val_samples, num_classes)
                                containing the validation class labels (optional).
            heuristic_type (str): The type of heuristic to use. Options: 'all', 'mean', 'sqrt',
                                'kolmogorov', 'fletcher_gloss', 'baum_haussler'.

        Returns:
            tuple: The number of neurons for the hidden layer.
        """
        heuristic_hidden_neurons = [10, 20, 50, 100, 200, 300]
        best_loss = np.inf
        best_neurons = None

        # Helper function to train and evaluate MLP for given number of neurons
        def evaluate_mlp(neurons):
            mlp_classifier = MLP(hidden_layer_sizes=(neurons,), learning_rate=self.learning_rate,
                                num_epochs=self.num_epochs, epochs_step=self.epochs_step,
                                loss='mse', shuffle_data=self.shuffle_data,
                                early_stopping=self.early_stopping)
            print(f'\n[heuristic_hidden_neurons][train]: {neurons} neurons\n')
            mlp_classifier.train(X, y, X_val, y_val)
            activations = mlp_classifier.forward_propagation(X)
            loss = mlp_classifier.compute_loss(y, activations[-1])
            return loss
        
        # Heuristic functions
        heuristic_functions = {
            'mean': lambda: int(np.mean([X.shape[1], y.shape[1]])),
            'sqrt': lambda: int(np.sqrt(X.shape[1] * y.shape[1])),
            'kolmogorov': lambda: int(2 * (X.shape[1]) + 1),
            'fletcher_gloss': lambda: int(np.sqrt((X.shape[1] * y.shape[1]) + 2)),
            'baum_haussler': lambda: int((X.shape[1] + y.shape[1]) / 2),
        }

        # If the heuristic type is 'all' or a specific heuristic, apply the corresponding heuristic
        heuristic_types_to_use = [heuristic_type] if heuristic_type != 'all' else heuristic_functions.keys()

        for h_type in heuristic_types_to_use:

            print(f'\n----------------------------------------------------------------')
            print(f'[heuristic_hidden_neurons][{h_type}] Start training')
            print(f'----------------------------------------------------------------')

            neurons = heuristic_functions[h_type]()

            if h_type == 'all':
                loss = evaluate_mlp(neurons)

                if loss < best_loss:
                    best_loss = loss
                    best_neurons = (neurons,)
            else:
                best_neurons = (neurons,) 

        print(f'\n[heuristic_hidden_neurons][best_neurons]: {best_neurons}\n')
        return best_neurons

    def _find_hidden_neurons_pca(self, X, y, max_hidden_neurons=None, variance_ratio=0.95):
        """
        Use PCA-based technique to estimate the number of hidden neurons.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                            containing the training samples.
            y (numpy.ndarray): The target matrix of shape (num_samples, num_classes)
                            containing the class labels.
            max_hidden_neurons (int, optional): Maximum number of hidden neurons to consider.
                                                If not provided, it will default to the minimum
                                                of num_features and num_classes.
            variance_ratio (float, optional): The maximum cumulative variance ratio to consider
                                              during PCA. Defaults to 0.95, which keeps 95% of the variance.

        Returns:
            int: The number of neurons for the hidden layer.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays.")
        
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must have 2 dimensions.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same.")

        num_features = X.shape[1]
        num_classes = y.shape[1]
        
        if max_hidden_neurons is None:
            max_hidden_neurons = max(num_features, num_classes)

        pca_cov = np.cov(X, rowvar=False)
        eigenvalues, _ = np.linalg.eig(pca_cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        best_neurons = np.argmax(cumulative_variance_ratio >= variance_ratio) + 1  # Keep 95% of the variance
        best_neurons = min(best_neurons, max_hidden_neurons)  # Cap the value at max_hidden_neurons

        print(f'\n[pca_hidden_neurons][best_neurons]: {best_neurons}\n')
        return (best_neurons,)


    def _exhaustive_search(self, X, y, X_val=None, y_val=None, step=None, min_hidden_neurons=None, max_hidden_neurons=None, hidden_neurons_options=None):
        """
        Perform an exhaustive search for the best number of hidden neurons.

        Parameters:
            X (numpy.ndarray): The feature matrix of shape (num_samples, num_features)
                                containing the training samples.
            y (numpy.ndarray): The target matrix of shape (num_samples, num_classes)
                                containing the class labels.
            X_val (numpy.ndarray): The feature matrix of shape (num_val_samples, num_features)
                                containing the validation samples (optional).
            y_val (numpy.ndarray): The target matrix of shape (num_val_samples, num_classes)
                                containing the validation class labels (optional).
            step (int): Step size for generating hidden_neurons_options using Method 3.
                        If not provided, the existing step value will be used. Default is 10.
            min_hidden_neurons (int): Minimum number of hidden neurons.
                                    If not provided, the existing min_hidden_neurons value will be used.
            max_hidden_neurons (int): Maximum number of hidden neurons.
                                    If not provided, the existing max_hidden_neurons value will be used.
            hidden_neurons_options (list or str): The list of hidden neurons to be used
                                                for the exhaustive search, or a string
                                                representing the method to generate it:
                                                'step_method', 'log_method', or 'len_method'.
                                                If not provided, the existing list will be used.

        Returns:
            tuple: The number of neurons for the hidden layer.
        """
        if min_hidden_neurons is None:
            min_hidden_neurons = min(X.shape[1], y.shape[1])

        if max_hidden_neurons is None:
            max_hidden_neurons = max(X.shape[1], y.shape[1])

        if step is None:
            step = 10

        # Method 1: Using range to generate integers from 1 to max_hidden_neurons
        len_method = tuple(range(1, max_hidden_neurons + 1))

        # Method 2: Using list comprehension to generate powers of 2 up to max_hidden_neurons
        log_method = [2**n for n in range(int(np.log2(max_hidden_neurons)) + 1)]

        # Method 3: Using range with parameters to generate integers within a specified range
        step_method = list(range(min_hidden_neurons, max_hidden_neurons + 1, step))

        if hidden_neurons_options is None:
            hidden_neurons_options = step_method
        elif hidden_neurons_options == 'step_method':
            hidden_neurons_options = step_method
        elif hidden_neurons_options == 'log_method':
            hidden_neurons_options = log_method
        elif hidden_neurons_options == 'len_method':
            hidden_neurons_options = len_method

        best_loss = np.inf
        best_neurons = None

        print('max_hidden_neurons', max_hidden_neurons)
        print('hidden_neurons_options', hidden_neurons_options)

        for neurons in hidden_neurons_options:
            mlp_classifier = MLP(hidden_layer_sizes=(neurons,), learning_rate=self.learning_rate,
                                 num_epochs=self.num_epochs, epochs_step=self.epochs_step,
                                 loss='mse', shuffle_data=self.shuffle_data,
                                 early_stopping=self.early_stopping)
            print(f'\n[exhaustive_search_hidden_neurons][train]: {neurons} neurons')
            mlp_classifier.train(X, y, X_val, y_val)
            activations = mlp_classifier.forward_propagation(X)
            loss = mlp_classifier.compute_loss(y, activations[-1])

            if loss < best_loss:
                best_loss = loss
                best_neurons = (neurons,)
        print(f'\n[exhaustive_search_hidden_neurons][best_neurons]: {neurons}\n')
        return best_neurons

    def softmax(self, x):
        """
        Apply the softmax function element-wise to the input array.

        Parameters:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: The array with softmax function applied element-wise.
        """
        expx = np.exp(x - np.max(x, axis=1, keepdims=True))
        return expx / np.sum(expx, axis=1, keepdims=True)
