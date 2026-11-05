import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        A = X.T @ X
        b = X.T @ y
        self.theta = np.linalg.solve(A, b)
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x = X[:, 1]
        n = x.shape[0]
        phi = np.ones((n, k + 1))
        for j in range(1, k + 1):
            phi[:, j] = x ** j
        return phi
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        x = X[:, 1]
        n = x.shape[0]
        phi = np.ones((n, k + 2))
        for j in range(1, k + 1):
            phi[:, j] = x ** j
        phi[:, k + 1] = np.sin(x)
        return phi
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return X @ self.theta
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        model = LinearModel()
        if sine:
            train_features = model.create_sin(k, train_x)
        else:
            train_features = model.create_poly(k, train_x)
        model.fit(train_features, train_y)
 
        if sine:
            plot_features = model.create_sin(k, plot_x)
        else:
            plot_features = model.create_poly(k, plot_x)
        plot_y = model.predict(plot_features)
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # Part (b): degree-3 polynomial regression
    run_exp(train_path, sine=False, ks=[3], filename='degree3_poly.png')
 
    # Part (c): degree-k polynomial regression
    run_exp(train_path, sine=False, ks=[3, 5, 10, 20], filename='degreek_poly.png')
 
    # Part (d): sine + polynomial feature map
    run_exp(train_path, sine=True, ks=[0, 1, 2, 3, 5, 10, 20], filename='degreek_sin.png')

    # Part (e): same polynomial feature map on the small dataset
    run_exp(small_path, sine=False, ks=[1, 2, 5, 10, 20], filename='small_poly.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
