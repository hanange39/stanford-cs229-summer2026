import numpy as np
import util
import matplotlib.pyplot as plt


def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train, y_train)

    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = clf.predict(x_eval)
    np.savetxt(save_path, y_pred)

    # Scatter plot
    plt.figure()
    plt.scatter(y_eval, y_pred, alpha=0.6)
    plt.xlabel('True counts')
    plt.ylabel('Predicted expected counts')
    plt.title('Poisson Regression: True vs Predicted Counts (Validation Set)')
    plt.plot([y_eval.min(), y_eval.max()],
             [y_eval.min(), y_eval.max()],
             'r--', linewidth=1, label='y = x')
    plt.legend()
    plt.savefig('poisson_valid_scatter.png')
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.
        Update the parameter by step_size * (sum of the gradient over examples)

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape

        # Initialize theta to the zero vector
        if self.theta is None:
            self.theta = np.zeros(dim)

        for i in range(self.max_iter):
            # h_theta(x) = E[y|x;theta] = exp(theta^T x), shape (n_examples,)
            h_x = np.exp(x.dot(self.theta))

            # Batch gradient of the log-likelihood
            gradient = x.T.dot(y - h_x)

            # Gradient ascent
            theta_new = self.theta + self.step_size * gradient

            # Stopping criterion
            if np.linalg.norm(theta_new - self.theta, ord=2) < self.eps:
                self.theta = theta_new
                if self.verbose:
                    print(f'Converged after {i + 1} iterations.')
                break

            self.theta = theta_new

            if self.verbose and i % 100000 == 0:
                # Average log-likelihood 
                ll = np.mean(y * (x.dot(self.theta)) - h_x)
                print(f'Iteration {i}: avg log-likelihood = {ll:.4f}')
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        # h_theta(x) = E[y|x;theta] = exp(theta^T x)
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***


if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')