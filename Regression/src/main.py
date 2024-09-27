from core.mlp import MLPRegressor
from core.multiple_linear_regression import MultipleLinearRegression
from utils.decorators import timer
from utils.evaluator import Evaluator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


@timer(repetitions=10)
def train_regression(regression, X, y):
    return regression.train(X, y)

def evaluate_classifier(y, y_pred, X, title):

    print('\n----------------------------------------------------------------------')
    print(f'[INFO][{title}] Result')

    # Create a new DataFrame to organize the output
    output_df = pd.DataFrame({
        'Actual Y': y,
        'Predicted Y': y_pred
    })

    # Display the output DataFrame
    print('\n----------------------------------------------------------------------')
    print(output_df)

    # Calculate evaluation metrics using Evaluator class
    mae = Evaluator.mean_absolute_error(y, y_pred)
    rmse = Evaluator.root_mean_square_error(y, y_pred)
    r2 = Evaluator.r_squared(y, y_pred)
    adjusted_r2 = Evaluator.adjusted_r_squared(y, y_pred, X.shape[1])
    mape = Evaluator.mean_absolute_percentage_error(y, y_pred)
    explained_variance = Evaluator.explained_variance_score(y, y_pred)

    Evaluator.plot_r_squared(y, y_pred,
                             title=f"{title} - R²",
                             save_path=f'src/results/{title}_r2.png')

    # Display the evaluation metrics
    print('\n----------------------------------------------------------------------')
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Square Error:", rmse)
    print("R-squared (R2):", r2)
    print("Adjusted R-squared:", adjusted_r2)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("Explained Variance Score:", explained_variance)
    print('----------------------------------------------------------------------')

def normalize_data(data):
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation for each feature.

    Parameters:
        data (numpy.ndarray): The input data as a NumPy array of shape (num_samples, num_features).

    Returns:
        numpy.ndarray: The normalized data as a NumPy array with the same shape as the input data.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def main() -> None:

    # -----------------------------------------------------------------------------------------------
    df = pd.read_excel('src/data/dataset.xlsx', engine='openpyxl')

    # Split the data into features (X) and target variable (y)
    X = df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
            'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
    y = df['Y house price of unit area']
    # -----------------------------------------------------------------------------------------------

    X_nparr = X.to_numpy() # Convert the Pandas DataFrame to a NumPy array
    y_nparr = y.to_numpy() # Convert the Pandas Series to a NumPy array

    X_nparr_norm = normalize_data(X_nparr)
    y_nparr_norm = normalize_data(y_nparr)

    X = X_nparr_norm
    y = y_nparr_norm


    # -----------------------------------------------------------------------------------------------
    # MLP Regression
    # -----------------------------------------------------------------------------------------------
    hidden_layer_sizes=(154,)
    learning_rate=0.068
    num_epochs=3000
    epochs_step=10
    shuffle_data=True
    # early_stopping={'patience': 40, 'delta': 1e-4}
    early_stopping=None

    # Create an instance of the Multiple Linear Regression class
    mlp_regression = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                              learning_rate=learning_rate,
                              num_epochs=num_epochs,
                              epochs_step=epochs_step,
                              shuffle_data=shuffle_data,
                              early_stopping=early_stopping)
    # -----------------------------------------------------------------------------------------------

    # -----------------------------------------------------------------------------------------------
    # Multiple Linear Regression
    # -----------------------------------------------------------------------------------------------
    mlr_regression = MultipleLinearRegression()
    # -----------------------------------------------------------------------------------------------


    # Train the MLP Regression
    train_regression(mlp_regression, X_nparr_norm, y_nparr_norm)
    mlp_y_pred = mlp_regression.predict(X_nparr_norm)
    mlp_r2 = Evaluator.r_squared(y_nparr_norm, mlp_y_pred)

    # Train the Multiple Linear Regression
    train_regression(mlr_regression, X, y)
    mlr_y_pred = mlr_regression.predict(X)
    mlr_r2 = Evaluator.r_squared(y, mlr_y_pred)

    # Combine the predicted values from all models
    all_y_pred = np.column_stack(( mlp_y_pred, mlr_y_pred))

    all_r2 = [mlp_r2, mlr_r2]

    all_models = ['MLP', 'MLR']


    _all_ = [[mlp_y_pred, 'MLP'], [mlr_y_pred, 'MLR']]


    [evaluate_classifier(y, x[0], X, x[1]) for x in _all_]


    # Set up the scatter plot
    plt.figure(figsize=(10, 8))
    
    for idx, y_pred in enumerate(all_y_pred.T):
        label = f"{all_models[idx]} (R²: {all_r2[idx]:.2f})" if all_r2[idx] is not None else f"Model {idx + 1}"
        plt.scatter(y, y_pred, label=label, alpha=0.5)

        # Calculate and plot the linear regression line
        regression_line = np.polyfit(y, y_pred, 1)
        regression_fn = np.poly1d(regression_line)
        plt.plot(y, regression_fn(y), label=f"Predição linear {idx + 1}", linewidth=2)

    plt.xlabel("Valor Verdadeiro")
    plt.ylabel("Valor Previsto")
    # plt.title("Predicted vs. True Values for Different Models")
    plt.legend()
    plt.grid(True)
    plt.show()


  

def plot_scatter_with_regression(r_squared, y_true, y_pred, title, ax):
    # Calculate regression line
    regression_line = np.polyfit(y_true, y_pred, 1)
    regression_fn = np.poly1d(regression_line)

    # Scatter plot and regression line
    ax.scatter(y_true, y_pred, label='Data', s=30, alpha=0.3)
    ax.plot(y_true, regression_fn(y_true), color='red', label='Regression Line', linewidth=1)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # Annotate R-squared value on the plot
    ax.annotate(f"R²: {r_squared:.2f}", xy=(0.83, 0.05), xycoords='axes fraction', fontsize=12)
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    main()