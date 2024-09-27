from AdalineSigmoid.core.adaline_sigmoid import AdalineSigmoid, AdalineSigmoidAutoencoder
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

def run_classifier(use_pca=False, use_autoencoder=False, n_components_percent=0.95):
    # Carregar e preparar os dados
    X, y = load_mnist(normalize=True)
    Y, lb = encode_labels(y)
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # Separar dados de validação
    X_train, X_val = X_train[:50000], X_train[50000:]
    Y_train, Y_val = Y_train[:50000], Y_train[50000:]

    if use_pca:
        X_train, X_test = apply_pca(X_train, X_test, n_components_percent)
        X_val, _ = apply_pca(X_val, X_val, n_components_percent)  # Aplicar PCA aos dados de validação

    # Iniciar a contagem do tempo
    start_time = time.time()

    # Configurar o classificador
    input_size = X_train.shape[1]
    hidden_size = 100
    target_digit = 5

    if use_autoencoder:
        adaline_epochs = 1000
        autoencoder_epochs = 20
        classifier = AdalineSigmoidAutoencoder(input_size, hidden_size, adaline_epochs=adaline_epochs, autoencoder_epochs=autoencoder_epochs, learning_rate=0.001)
    else:
        classifier = AdalineSigmoid(learning_rate=0.001, n_iterations=1000)
    
    # Preparar dados para treinamento (classificação binária)
    Y_train_binary = (Y_train[:, target_digit] == 1).astype(int)
    Y_val_binary = (Y_val[:, target_digit] == 1).astype(int)
    
    # Treinar o classificador
    if use_autoencoder:
        classifier.train(X_train, Y_train_binary, X_val, Y_val_binary)
    else:
        classifier.fit(X_train, Y_train_binary, X_val=X_val, y_val=Y_val_binary)

    # Fazer previsões
    y_pred = classifier.predict(X_test)

    # Finalizar a contagem do tempo
    end_time = time.time()
    execution_time = end_time - start_time

    y_test_binary = (Y_test[:, target_digit] == 1).astype(int)

    # Calcular e exibir a acurácia
    accuracy = np.mean(y_pred == y_test_binary)
    error_rate = 1 - accuracy

    # Exibir resultados
    print(f"Resultados para o dígito {target_digit}:")
    print(f"Acurácia: {accuracy * 100:.2f}%")
    print(f"Taxa de erro: {error_rate * 100:.2f}%")
    print(f"Tempo de execução: {execution_time:.2f} segundos")

    # Plotar e salvar a matriz de confusão
    pca_str = "com_pca" if use_pca else "sem_pca"
    ae_str = "com_autoencoder" if use_autoencoder else "sem_autoencoder"
    save_path = f'AdalineSigmoid/results/confusion_matrix_{pca_str}_{ae_str}_digit_{target_digit}.png'
    plot_confusion_matrix(y_test_binary, y_pred, ['Não ' + str(target_digit), str(target_digit)], save_path)

    # Plotar a curva de custo (se disponível)
    if hasattr(classifier, 'costs'):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, len(classifier.costs) + 1), classifier.costs)
        plt.xlabel('Épocas')
        plt.ylabel('Custo')
        plt.title(f'Adaline Sigmoid - Custo por Época (Dígito {target_digit})')
        cost_path = f'AdalineSigmoid/results/cost_curve_{pca_str}_{ae_str}_digit_{target_digit}.png'
        plt.savefig(cost_path)
        plt.close()

if __name__ == "__main__":
    # # Executar Adaline Sigmoid sem PCA e sem Autoencoder
    # print("Executando Adaline Sigmoid sem PCA e sem Autoencoder:")
    # run_classifier(use_pca=False, use_autoencoder=False)

    # # Executar Adaline Sigmoid com PCA e sem Autoencoder
    # print("\nExecutando Adaline Sigmoid com PCA e sem Autoencoder:")
    # run_classifier(use_pca=True, use_autoencoder=False, n_components_percent=0.95)

    # Executar Adaline Sigmoid sem PCA e com Autoencoder
    print("\nExecutando Adaline Sigmoid sem PCA e com Autoencoder:")
    run_classifier(use_pca=False, use_autoencoder=True)

