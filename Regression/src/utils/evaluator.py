import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


class Evaluator:
    """
    Evaluator

    This class provides methods to evaluate the performance of a classifier using different metrics.

    Methods:
        accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            Calculate the accuracy of the classifier.

        class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
            Calculate the accuracy for each class.

        confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
            Compute the confusion matrix.

        plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix Heatmap', save_path: str = None) -> None:
            Plot the confusion matrix as a heatmap.
    """

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)

    @staticmethod
    def class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        class_accuracy = np.zeros(num_classes)
        for i in range(num_classes):
            class_mask = y_true == i
            class_accuracy[i] = np.mean(y_pred[class_mask] == i)
        return class_accuracy

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
        return np.histogram2d(y_true, y_pred, bins=num_classes)[0].astype(int)

    @staticmethod
    def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], title: str = 'Confusion Matrix Heatmap', save_path: str = None) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    @staticmethod
    def plot_r_squared(y_true: np.ndarray, y_pred: np.ndarray, title: str = "R²", save_path: str = None) -> None:
        r_squared = Evaluator.r_squared(y_true, y_pred)

        # Calculate regression line
        regression_line = np.polyfit(y_true, y_pred, 1)
        regression_fn = np.poly1d(regression_line)

        # Plot scatter plot and regression line
        plt.scatter(y_true, y_pred, label='Data', s=30, alpha=0.3)
        plt.plot(y_true, regression_fn(y_true), color='red', label='Regression Line', linewidth=1)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(title)

        # Annotate R-squared value on the plot
        plt.annotate(f"R²: {r_squared:.2f}", xy=(0.83, 0.05), xycoords='axes fraction', fontsize=12)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    @staticmethod
    def adjusted_r_squared(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
        n = len(y_true)
        r2 = Evaluator.r_squared(y_true, y_pred)
        return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    
