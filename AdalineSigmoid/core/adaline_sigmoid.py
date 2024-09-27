import numpy as np
from MNIST_utils.autoencoder import Autoencoder

class AdalineSigmoid:
    def __init__(self, learning_rate=0.01, n_iterations=100, random_state=1, lambda_reg=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.costs = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def fit(self, X, y, X_val=None, y_val=None):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.bias = np.float_(0.)
        
        best_val_cost = float('inf')
        patience = 10
        no_improve = 0
        
        for _ in range(self.n_iterations):
            net_input = self.net_input(X)
            output = self.sigmoid(net_input)
            errors = y - output
            
            self.weights += self.learning_rate * (X.T.dot(errors) - self.lambda_reg * self.weights)
            self.bias += self.learning_rate * np.sum(errors)
            
            output_clipped = np.clip(output, 1e-15, 1 - 1e-15)
            cost = -np.mean(y * np.log(output_clipped) + (1 - y) * np.log(1 - output_clipped))
            cost += 0.5 * self.lambda_reg * np.sum(self.weights ** 2)
            self.costs.append(cost)
            
            # Early stopping
            if X_val is not None and y_val is not None:
                val_output = self.sigmoid(self.net_input(X_val))
                val_output_clipped = np.clip(val_output, 1e-15, 1 - 1e-15)
                val_cost = -np.mean(y_val * np.log(val_output_clipped) + (1 - y_val) * np.log(1 - val_output_clipped))
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at iteration {_}")
                    break
        
        return self

    def predict(self, X):
        return (self.sigmoid(self.net_input(X)) >= 0.5).astype(int)
    
class AdalineSigmoidAutoencoder:
    def __init__(self, input_size, hidden_size, adaline_epochs=100, autoencoder_epochs=200, learning_rate=0.01):
        self.autoencoder = Autoencoder(input_size, hidden_size, learning_rate=learning_rate, epochs=autoencoder_epochs, patience=5, min_delta=0.001)
        self.adaline = AdalineSigmoid(learning_rate=learning_rate, n_iterations=int(adaline_epochs))

    def train(self, X, y, X_val, y_val):
        self.autoencoder.train(X, X_val, batch_size=32)
        encoded_X = self.autoencoder.encode(X)
        encoded_X_val = self.autoencoder.encode(X_val)
        self.adaline.fit(encoded_X, y, X_val=encoded_X_val, y_val=y_val)

    def predict(self, X):
        encoded_X = self.autoencoder.encode(X)
        return self.adaline.predict(encoded_X)