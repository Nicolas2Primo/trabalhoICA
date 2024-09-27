import time
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    return x * (1 - x)

class Autoencoder:
    def __init__(self, input_size, hidden_size, learning_rate=0.01, epochs=100, patience=5, min_delta=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, input_size) * 0.01
        self.bias2 = np.zeros((1, input_size))

    def forward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, output):
        output_error = X - output
        d_output = output_error * sigmoid_derivative(output)
        
        hidden_error = np.dot(d_output, self.weights2.T)
        d_hidden = hidden_error * sigmoid_derivative(self.hidden)
        
        self.weights2 += self.learning_rate * np.dot(self.hidden.T, d_output)
        self.bias2 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, d_hidden)
        self.bias1 += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, X, X_val, batch_size=32):
        n_samples = X.shape[0]
        n_batches = (n_samples - 1) // batch_size + 1
        start_time = time.time()
        
        best_val_loss = float('inf')
        no_improve_count = 0

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                output = self.forward(batch)
                self.backward(batch, output)
                
                batch_loss = np.mean((batch - output) ** 2)
                epoch_loss += batch_loss

            avg_epoch_loss = epoch_loss / n_batches
            
            # Calcular perda de validação
            val_output = self.forward(X_val)
            val_loss = np.mean((X_val - val_output) ** 2)

            elapsed_time = time.time() - start_time

            print(f"Época {epoch+1}/{self.epochs} - Perda Treino: {avg_epoch_loss:.4f} - Perda Validação: {val_loss:.4f} - Tempo: {elapsed_time:.2f}s")

            # Early stopping
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= self.patience:
                print(f"Early stopping ativado na época {epoch+1}")
                break

        print(f"Treinamento do Autoencoder concluído em {elapsed_time:.2f} segundos")

    def encode(self, X):
        return sigmoid(np.dot(X, self.weights1) + self.bias1)