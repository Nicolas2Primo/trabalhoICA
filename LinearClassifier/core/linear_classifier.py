import numpy as np

class LinearClassifier:
    """
    Classe para o Classificador Linear de Mínimos Quadrados.

    Atributos:
        W (np.ndarray ou None): Os pesos aprendidos para o classificador.
        regularization (str ou None): Tipo de regularização ('l2' ou None).
        lambda_value (float): Parâmetro de regularização.

    Métodos:
        fit(X: np.ndarray, Y: np.ndarray) -> None:
            Ajusta o modelo aos dados de treinamento.

        predict(X: np.ndarray) -> np.ndarray:
            Faz previsões usando o classificador treinado.
    """

    def __init__(self, lambda_value: float = 0.0):
        """
        Inicializa o classificador.

        Parâmetros:
        - lambda_value: Parâmetro de regularização.
        """
        self.W = None
        self.lambda_value = lambda_value

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Ajusta o modelo aos dados de treinamento.

        Parâmetros:
        - X: Matriz de entrada de dimensão (n_amostras, n_características).
        - Y: Matriz de rótulos one-hot de dimensão (n_amostras, n_classes).

        Retorna:
            None
        """
        # Adicionar o termo de bias
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Calcular os pesos usando a pseudoinversa
        self.W = np.linalg.pinv(X_bias) @ Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz previsões para os dados de entrada.

        Parâmetros:
        - X: Matriz de entrada de dimensão (n_amostras, n_características).

        Retorna:
        - y_pred: Vetor de previsões (n_amostras,).
        """
        # Adicionar o termo de bias
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        # Calcular as pontuações
        Y_scores = X_bias @ self.W

        if Y_scores.shape[1] == 1:
            # Para classificação binária, retornar rótulos 0 ou 1 baseados em um limiar
            return (Y_scores >= 0.5).astype(int).flatten()
        else:
            # Para classificação multiclasse, retornar a classe com a maior pontuação
            return np.argmax(Y_scores, axis=1)

