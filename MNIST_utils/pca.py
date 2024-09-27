import numpy as np

class PCA:
    def __init__(self, n_components_percent):
        """
        Inicializa a classe PCA com a porcentagem desejada de variância a ser retida.

        Parâmetros:
            n_components_percent (float): A porcentagem da variância total a ser retida
                                          nos dados transformados.

        Atributos:
            n_components_percent (float): A porcentagem especificada de variância a reter.
            components (numpy.ndarray): Os componentes principais (autovetores) obtidos
                                        após ajustar o PCA aos dados.
            mean (numpy.ndarray): A média dos dados usada para centralizar os dados.
            total_variance (float): A variância total dos dados usada para calcular
                                    a porcentagem de variância explicada.
        """
        self.n_components_percent = n_components_percent
        self.components = None
        self.mean = None
        self.total_variance = None

    def fit(self, X):
        """
        Ajusta o modelo PCA aos dados de entrada.

        Parâmetros:
            X (numpy.ndarray): Os dados de entrada com forma (n_amostras, n_características).

        Retorna:
            None
        """
        # Calcula a média dos dados
        self.mean = np.mean(X, axis=0)
        centered_data = X - self.mean

        # Calcula a matriz de covariância
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Calcula os autovalores e autovetores da matriz de covariância
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Ordena os autovalores e autovetores correspondentes em ordem decrescente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Calcula a variância total
        self.total_variance = np.sum(sorted_eigenvalues)

        # Calcula o número de componentes que explicam a porcentagem dada de variância
        cumulative_variance = np.cumsum(sorted_eigenvalues) / self.total_variance
        n_components = np.argmax(cumulative_variance >= self.n_components_percent) + 1

        # Seleciona os n_components principais autovetores
        self.components = sorted_eigenvectors[:, :n_components]

    def transform(self, X):
        """
        Transforma os dados de entrada usando o modelo PCA ajustado.

        Parâmetros:
            X (numpy.ndarray): Os dados de entrada com forma (n_amostras, n_características).

        Retorna:
            numpy.ndarray: Os dados transformados com forma (n_amostras, n_componentes),
                           onde n_componentes é o número de componentes principais
                           especificados para reter a porcentagem desejada de variância.
        """
        # Projeta os dados nos autovetores selecionados
        centered_data = X - self.mean
        transformed_data = np.dot(centered_data, self.components)
        return transformed_data
