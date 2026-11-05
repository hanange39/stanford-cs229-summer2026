import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    n, dim = x.shape
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    group_idxs = np.array_split(np.random.permutation(n), K)
    mu = [np.mean(x[idxs], axis=0) for idxs in group_idxs]
    sigma = [np.cov(x[idxs].T) for idxs in group_idxs]

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full((K,), 1.0 / K)

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.full((n, K), 1.0 / K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000
    n = x.shape[0]

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        w, ll = compute_w_and_ll(x, phi, mu, sigma)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            w_j = w[:, j] # shape (n,)
            w_j_sum = np.sum(w_j)
 
            phi[j] = w_j_sum / n
 
            mu[j] = (w_j[:, None] * x).sum(axis=0) / w_j_sum
 
            diff = x - mu[j] # shape (n, dim)
            sigma[j] = (w_j[:, None, None] *
                        np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / w_j_sum
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        _, ll = compute_w_and_ll(x, phi, mu, sigma)
        it += 1
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000
    n = x.shape[0]              
    n_tilde = x_tilde.shape[0]  
    z_tilde_flat = z_tilde.squeeze().astype(int)

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w, ll = compute_semi_supervised_ll(x, x_tilde, z_tilde, phi, mu, sigma, alpha)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            w_j = w[:, j] # shape (n,)
            tilde_w_j = alpha * (z_tilde_flat == j).astype(float)  # shape (n_tilde,)
 
            denom_j = np.sum(w_j) + np.sum(tilde_w_j)
 
            phi[j] = denom_j / (n + alpha * n_tilde)
 
            mu[j] = (w_j[:, None] * x).sum(axis=0)
            mu[j] += (tilde_w_j[:, None] * x_tilde).sum(axis=0)
            mu[j] /= denom_j
 
            diff = x - mu[j] # shape (n, dim)
            diff_tilde = x_tilde - mu[j] # shape (n_tilde, dim)
 
            sigma[j] = (w_j[:, None, None] *
                        np.einsum('ni,nj->nij', diff, diff)).sum(axis=0)
            sigma[j] += (tilde_w_j[:, None, None] *
                         np.einsum('ni,nj->nij', diff_tilde, diff_tilde)).sum(axis=0)
            sigma[j] /= denom_j
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        _, ll = compute_semi_supervised_ll(x, x_tilde, z_tilde, phi, mu, sigma, alpha)
        it += 1
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
# Helper functions
def gaussian_pdf(x_row, mu_j, sigma_j):
    """Multivariate Gaussian density"""
    d = x_row.shape[0]
    diff = x_row - mu_j
    sigma_inv = np.linalg.inv(sigma_j)
    sign, logdet = np.linalg.slogdet(sigma_j)
    log_norm_const = -0.5 * (d * np.log(2 * np.pi) + logdet)
    log_exponent = -0.5 * diff.T @ sigma_inv @ diff
    return np.exp(log_norm_const + log_exponent)

def compute_w_and_ll(x, phi, mu, sigma):
    """Compute the E-step responsibilities w[i, j] and the log-likelihood"""
    n = x.shape[0]
    joint = np.zeros((n, K))  # joint[i, j] = p(x^(i) | z^(i)=j) * phi[j]
    for i in range(n):
        for j in range(K):
            joint[i, j] = gaussian_pdf(x[i], mu[j], sigma[j]) * phi[j]
 
    denom = np.sum(joint, axis=1, keepdims=True)  # p(x^(i)) for each i
    w = joint / denom
    ll = np.sum(np.log(denom))
    return w, ll

def compute_semi_supervised_ll(x, x_tilde, z_tilde, phi, mu, sigma, alpha):
    """Compute the E-step responsibilities w for the unlabeled data and the
    semi-supervised log-likelihood"""
    n = x.shape[0]
    n_tilde = x_tilde.shape[0]

    # unsupervised part (same as compute_w_and_ll)
    joint = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            joint[i, j] = gaussian_pdf(x[i], mu[j], sigma[j]) * phi[j]
    denom = np.sum(joint, axis=1, keepdims=True)
    w = joint / denom
    ll_unsup = np.sum(np.log(denom))
 
    # supervised part (z_tilde is observed, so no summation over z needed)
    ll_sup = 0.0
    for i in range(n_tilde):
        j = int(z_tilde[i, 0])
        ll_sup += np.log(gaussian_pdf(x_tilde[i], mu[j], sigma[j]) * phi[j])
 
    ll = ll_unsup + alpha * ll_sup
    return w, ll
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'), fontsize=18)
    plt.xlabel('x_1', fontsize=14)
    plt.ylabel('x_2', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
