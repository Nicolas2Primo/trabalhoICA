from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def encode_labels(y):
    """
    Codifica os rótulos usando codificação one-hot.

    Parameters:
    - y: Vetor de rótulos.

    Returns:
    - Y: Matriz de rótulos codificados one-hot.
    - lb: Objeto LabelBinarizer ajustado.
    """
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    return Y, lb

def split_data(X, Y, test_size=1/7, random_state=42):
    """
    Divide os dados em conjuntos de treinamento e teste.

    Parameters:
    - X: Matriz de características.
    - Y: Matriz de rótulos (pode ser one-hot ou não).
    - test_size: Proporção do conjunto de teste.
    - random_state: Semente aleatória.

    Returns:
    - X_train, X_test, Y_train, Y_test: Dados divididos.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test