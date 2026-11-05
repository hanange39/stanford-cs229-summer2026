import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)

    # Get MSE value on the validation set
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    mse = np.mean((y_pred - y_eval) ** 2)
    print(f'MSE={mse}')

    # Plot validation predictions on top of training set
    plt.figure()
    plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2, label='training')
    plt.plot(x_eval[:, -1], y_pred, 'ro', linewidth=2, label='predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('lwr_plot.png')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            x_query = x[i]  # shape (n,)

            # Compute weights: w^(i) = exp(-||x^(i) - x||^2 / (2*tau^2))
            diffs = self.x - x_query   # (n_train, n)
            sq_dists = np.sum(diffs ** 2, axis=1)   # (n_train,)
            w = np.exp(-sq_dists / (2 * self.tau ** 2))  # (n_train,)

            W = np.diag(w)  # (n_train, n_train)

            # Weighted normal equation: theta = (X^T W X)^-1 X^T W y
            XtW = self.x.T @ W   # (n, n_train)
            theta = np.linalg.solve(XtW @ self.x, XtW @ self.y)  # (n,)

            y_pred[i] = x_query @ theta

        return y_pred
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
