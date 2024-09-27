from LinearClassifier.core.linear_classifier import LinearClassifier
from MNIST_utils.pca import PCA
from MNIST_utils.data_loader import load_mnist
from MNIST_utils.preprocessing import encode_labels, split_data
from MNIST_utils.visualization import plot_confusion_matrix
import numpy as np
import time

def apply_pca(X_train, X_test, n_components_percent):
    pca = PCA(n_components_percent)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def run_classifier(use_pca=False, n_components_percent=0.95):
    # Carregar e preparar os dados
    X, y = load_mnist(normalize=True)
    Y, lb = encode_labels(y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    if use_pca:
        X_train, X_test = apply_pca(X_train, X_test, n_components_percent)

    # Iniciar a contagem do tempo
    start_time = time.time()

    # Treinar o classificador
    classifier = LinearClassifier(lambda_value=1e-5)
    classifier.fit(X_train, Y_train)

    # Fazer previsões
    y_pred = classifier.predict(X_test)

    # Finalizar a contagem do tempo
    end_time = time.time()
    execution_time = end_time - start_time

    y_test = lb.inverse_transform(Y_test)

    # Calcular e exibir a acurácia
    accuracy = np.mean(y_pred == y_test)
    error_rate = 1 - accuracy

    # Exibir resultados
    print(f"Acurácia: {accuracy * 100:.2f}%")
    print(f"Taxa de erro: {error_rate * 100:.2f}%")
    print(f"Tempo de execução: {execution_time:.2f} segundos")

    # Plotar e salvar a matriz de confusão
    pca_str = "com_pca" if use_pca else "sem_pca"
    save_path = f'LinearClassifier/results/confusion_matrix_{pca_str}.png'
    plot_confusion_matrix(y_test, y_pred, lb.classes_, save_path)

if __name__ == "__main__":
    # Executar sem PCA
    print("Executando classificador sem PCA:")
    run_classifier(use_pca=False)

    # Executar com PCA
    print("\nExecutando classificador com PCA:")
    run_classifier(use_pca=True, n_components_percent=0.95)  # Mantendo 95% da variância