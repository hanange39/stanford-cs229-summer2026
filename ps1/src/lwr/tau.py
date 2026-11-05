import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)

    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = None
    best_mse = float('inf')

    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_valid)
        mse = np.mean((y_pred - y_valid) ** 2)
        print(f'tau={tau}: validation MSE={mse}')

        if mse < best_mse:
            best_mse = mse
            best_tau = tau

        # Plot data for this tau
        plt.figure()
        plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2, label='training')
        plt.plot(x_valid[:, -1], y_pred, 'ro', linewidth=2, label='predictions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Validation Set (Tau={tau}, MSE={mse})')
        plt.legend()
        plt.savefig(f'lwr_tau_{tau}.png')
        plt.close()

    print(f'Best tau on validation set: {best_tau} (MSE={best_mse})')

    # Fit a LWR model with the best tau value
    clf = LocallyWeightedLinearRegression(best_tau)
    clf.fit(x_train, y_train)

    # Run on the test set to get the MSE value
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_test_pred = clf.predict(x_test)
    test_mse = np.mean((y_test_pred - y_test) ** 2)
    print(f'Test MSE using best tau={best_tau}: {test_mse}')

    # Save predictions to pred_path
    np.savetxt(pred_path, y_test_pred)

    # Plot data
    plt.figure()
    plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2, label='training')
    plt.plot(x_test[:, -1], y_test_pred, 'ro', linewidth=2, label='predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Test Set (Tau={best_tau}, Test MSE={test_mse:.4f})')
    plt.legend()
    plt.savefig('lwr_best_tau_test.png')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')