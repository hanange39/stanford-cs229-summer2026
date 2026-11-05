import numpy as np
import util
import sys
from random import random

sys.path.append('../logreg_stability')

### NOTE : You need to complete logreg implementation first! If so, make sure to set the regularization weight to 0.
from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1


def compute_metrics(y_true, y_pred):
    """Compute accuracy, balanced accuracy, and per-class accuracies.

    Args:
        y_true: (n,) array of true labels in {0, 1}.
        y_pred: (n,) array of predicted labels in {0, 1}.

    Returns:
        A, A_bar, A0, A1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    A = (TP + TN) / (TP + TN + FP + FN)
    A1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    A0 = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    A_bar = 0.5 * (A0 + A1)

    return A, A_bar, A0, A1


def print_metrics(name, y_true, y_pred):
    A, A_bar, A0, A1 = compute_metrics(y_true, y_pred)
    print(f'--- {name} ---')
    print(f'Accuracy (A):          {A:.4f}')
    print(f'Balanced accuracy (A~): {A_bar:.4f}')
    print(f'Class 0 accuracy (A0):  {A0:.4f}')
    print(f'Class 1 accuracy (A1):  {A1:.4f}')
    print()


def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***

    # Load data 
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)

    # Part (b): Vanilla logistic regression
    model_naive = LogisticRegression()
    model_naive.fit(x_train, y_train)

    probs_naive = model_naive.predict(x_val) # predicted P(y=1|x)
    preds_naive = (probs_naive >= 0.5).astype(int)

    np.savetxt(output_path_naive, probs_naive)

    print_metrics('Vanilla logistic regression (Part b)', y_val, preds_naive)

    # Plot validation set + decision boundary for vanilla model
    util.plot(x_val, y_val, model_naive.theta,
              save_path='imbalanced_naive.png')
  
    # Part (d): Upsampling the minority class
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    repeat_factor = int(round(1 / kappa))
    x_train_up = np.concatenate(
        [x_train[neg_idx]] + [x_train[pos_idx]] * repeat_factor, axis=0)
    y_train_up = np.concatenate(
        [y_train[neg_idx]] + [y_train[pos_idx]] * repeat_factor, axis=0)

    model_upsampling = LogisticRegression()
    model_upsampling.fit(x_train_up, y_train_up)

    probs_upsampling = model_upsampling.predict(x_val)
    preds_upsampling = (probs_upsampling >= 0.5).astype(int)

    np.savetxt(output_path_upsampling, probs_upsampling)

    print_metrics('Upsampled logistic regression (Part d)', y_val, preds_upsampling)

    # Plot validation set + decision boundary for upsampled model
    util.plot(x_val, y_val, model_upsampling.theta,
              save_path='imbalanced_upsampling.png')

    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')