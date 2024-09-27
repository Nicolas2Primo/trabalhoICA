import numpy as np
import pandas as pd
import os

def load_mnist(normalize=True):
    """
    Carrega o conjunto de dados MNIST a partir de arquivos CSV locais.

    Parameters:
    - normalize: Se True, normaliza os dados para o intervalo [0,1].

    Returns:
    - X: Matriz de características.
    - y: Vetor de rótulos.
    """
    print("Carregando o conjunto de dados MNIST...")
    
    # Definir os caminhos para os arquivos CSV usando caminho relativo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Subir dois níveis
    train_path = os.path.join(parent_dir, 'MNIST_data', 'mnist_train.csv')
    test_path = os.path.join(parent_dir, 'MNIST_data', 'mnist_test.csv')
    
    # Verificar se os arquivos existem
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Arquivos MNIST não encontrados em {os.path.join(parent_dir, 'MNIST_data')}")
    
    # Carregar os dados de treino e teste
    train_data = pd.read_csv(train_path, header=0)  # Assumindo que há um cabeçalho
    test_data = pd.read_csv(test_path, header=0)    # Assumindo que há um cabeçalho
    
    # Combinar os dados de treino e teste
    data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    
    # Separar as características (X) e os rótulos (y)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    # Converter rótulos para inteiros
    y = y.astype(int)
    
    if normalize:
        X = X.astype(float) / 255.0
    
    return X, y