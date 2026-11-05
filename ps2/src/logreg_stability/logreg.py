import numpy as np
import util


def main(train_path, save_path, reg_lambda=0.0):
    """Problem: Logistic regression with gradient descent.

    Args:
        train_path: Path to CSV file containing dataset for training.
        save_path: Path to save outputs; visualizations, predictions, etc.
        reg_lambda: L2 regularization strength. 0 = no regularization (part a).
    """
    x_train, y_train = util.load_csv(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression(reg_lambda=reg_lambda)
    clf.fit(x_train, y_train)
    print(f'Final theta (lambda={reg_lambda}): {clf.theta}')

    # Plot decision boundary on top of training set
    suffix = f'_lambda{reg_lambda}' if reg_lambda > 0 else ''
    plot_path = save_path.replace('.txt', f'{suffix}.png')
    util.plot(x_train, y_train, clf.theta, plot_path)

    # Save predictions
    p_train = clf.predict(x_train)
    np.savetxt(save_path, p_train)
    # *** END CODE HERE ***
    return clf.theta


class LogisticRegression:
    """Logistic regression using gradient descent.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=1, max_iter=100000, eps=1e-5,
                 theta_0=None, verbose=True, reg_lambda=0.0):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
            reg_lambda: L2 regularization strength (lambda). 0 = no regularization.
        """
        self.theta = theta_0
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.reg_lambda = reg_lambda

        # *** START CODE HERE ***
        self.loss_eps = 1e-5
        # *** END CODE HERE ***

    def fit(self, x, y):
        """Run gradient descent to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape

        if self.theta is None:
            self.theta = np.zeros(dim)

        for k in range(self.max_iter):
            theta_prev = np.copy(self.theta)

            # h_theta(x) = g(theta^T x), shape (n_examples,)
            h = self._sigmoid(x.dot(self.theta))

            # Gradient of J(theta)
            grad = (1.0 / n_examples) * x.T.dot(h - y) + self.reg_lambda * self.theta

            self.theta -= self.learning_rate * grad

            if self.verbose and k % 1000 == 0:
                h_safe = self._sigmoid(x.dot(self.theta))
                loss = -np.mean(
                    y * np.log(h_safe + self.loss_eps) +
                    (1 - y) * np.log(1 - h_safe + self.loss_eps)
                ) + 0.5 * self.reg_lambda * np.sum(self.theta ** 2)
                print(f'Iteration {k}: loss = {loss:.6f}')

            # Convergence check
            if np.sum(np.abs(self.theta - theta_prev)) < self.eps:
                if self.verbose:
                    print(f'Converged after {k + 1} iterations.')
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return self._sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***

    def _sigmoid(self, z): # Helper
        # Avoid overflow in exp() for very negative/positive z by splitting
        # into two branches.
        out = np.zeros_like(z, dtype=float)
        pos_mask = z >= 0
        neg_mask = ~pos_mask

        out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
        exp_z = np.exp(z[neg_mask])
        out[neg_mask] = exp_z / (1.0 + exp_z)

        return out


if __name__ == '__main__':
    print('==== Training model on dataset A (no regularization) ====')
    theta_a = main(train_path='ds1_a.csv',
                    save_path='logreg_pred_a.txt')

    print('\n==== Training model on data et B (no regularization) ====')
    theta_b = main(train_path='ds1_b.csv',
                    save_path='logreg_pred_b.txt')

    print('\n==== Training model on dataset A (L2 regularization, lambda=0.01) ====')
    theta_a_reg = main(train_path='ds1_a.csv',
                        save_path='logreg_pred_a.txt',
                        reg_lambda=0.01)

    print('\n==== Training model on dataset B (L2 regularization, lambda=0.01) ====')
    theta_b_reg = main(train_path='ds1_b.csv',
                        save_path='logreg_pred_b.txt',
                        reg_lambda=0.01)

    print('\n==== Comparison ====')
    print(f'Dataset A: theta (no reg) = {theta_a}')
    print(f'Dataset A: theta (reg)    = {theta_a_reg}')
    print(f'Dataset B: theta (no reg) = {theta_b}')
    print(f'Dataset B: theta (reg)    = {theta_b_reg}')