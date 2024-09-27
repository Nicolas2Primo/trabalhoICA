import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression using the method of least squares.

    This class implements multiple linear regression, a statistical technique used to model the relationship 
    between multiple independent variables (features) and a dependent variable (target).
    """

    def __init__(self, alpha=0):
        """
        Initializes the MultipleLinearRegression class.

        Attributes:
            alpha (float): The regularization parameter. Default is 0, which corresponds to ordinary least squares.
            coefficients (numpy.ndarray): The learned coefficients of the linear regression model. The first element is
                                         the constant term (intercept), and the remaining elements are the coefficients 
                                         associated with each feature.
            scaler (numpy.ndarray): The mean of each feature used for feature scaling during training.
        """
        self.alpha = alpha
        self.coefficients = None
        self.scaler = None

    def train(self, X, y):
        """
        Train the multiple linear regression model to the given data using the method of least squares.

        Args:
            X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): The target variable vector of shape (n_samples,).
        """
        # Feature scaling using mean normalization
        self.scaler = np.mean(X, axis=0)
        X_scaled = (X - self.scaler) / np.std(X, axis=0)

        # Adding a column of 1's to represent the constant term (b0)
        X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

        # Calculating coefficients using the least squares formula
        n_features = X_scaled.shape[1]
        regularization_term = self.alpha * np.eye(n_features)
        regularization_term[0, 0] = 0  # Don't regularize the intercept term
        XtX_inv = np.linalg.inv(X_scaled.T.dot(X_scaled) + regularization_term)
        self.coefficients = XtX_inv.dot(X_scaled.T).dot(y)

    def predict(self, X):
        """
        Makes predictions using the learned linear regression model.

        Args:
            X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The predicted target variable vector of shape (n_samples,).
        """
        # Feature scaling using the scaler learned during training
        X_scaled = (X - self.scaler) / np.std(X, axis=0)

        # Adding a column of 1's to represent the constant term (b0)
        X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

        # Calculating predictions using the learned coefficients
        return X_scaled.dot(self.coefficients)

    def get_coefficients(self):
        """
        Returns the learned coefficients of the linear regression model.

        Returns:
            numpy.ndarray: The coefficients of the linear regression model.
        """
        return self.coefficients

    def get_intercept(self):
        """
        Returns the learned intercept (constant term) of the linear regression model.

        Returns:
            float: The intercept (constant term) of the linear regression model.
        """
        return self.coefficients[0]
