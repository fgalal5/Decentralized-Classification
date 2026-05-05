import os
import numpy as np
import matplotlib.pyplot as plt


############################################
# Data Loaders
############################################

def prepare_synthetic_data():
    '''
    Load the synthetic 2-D dataset used for visualization and testing.

    Data files:
        Data/synthetic/synthetic_X.csv
        Data/synthetic/synthetic_Y.csv

    The dataset has 1500 samples (2 features per sample) and 3 classes.
    The split is:
        800 training samples
        200 validation samples
        500 testing samples

    Returns:
        data (dict): containing
            'trainX', 'trainY', 'valX', 'valY', 'testX', 'testY'
    '''
    data = dict()
    X = np.loadtxt('Data/synthetic/synthetic_X.csv', delimiter=',').reshape(1500, 2)
    Y = np.loadtxt('Data/synthetic/synthetic_Y.csv', delimiter=',')

    data['trainX'] = X[:800]
    data['valX']   = X[800:1000]
    data['testX']  = X[1000:]
    data['trainY'] = Y[:800]
    data['valY']   = Y[800:1000]
    data['testY']  = Y[1000:]
    return data


def prepare_mnist_data():
    '''
    Load the reduced Fashion-MNIST dataset (grayscale, flattened 28×28 images).

    Data files:
        Data/reduced_fashion_mnist/mnist_X.csv
        Data/reduced_fashion_mnist/mnist_Y.csv

    The dataset has 1500 samples and 3 classes (0: T-shirt/top, 3: Dress, 4: Coat).
    Each sample is a flattened 784-dimensional vector.

    The split is:
        800 training samples
        200 validation samples
        500 testing samples

    Returns:
        data (dict): containing
            'trainX', 'trainY', 'valX', 'valY', 'testX', 'testY'
    '''
    data = dict()
    X = np.loadtxt('Data/reduced_fashion_mnist/mnist_X.csv', delimiter=',').reshape(1500, 784)
    Y = np.loadtxt('Data/reduced_fashion_mnist/mnist_Y.csv', delimiter=',')

    data['trainX'] = X[:800]
    data['valX']   = X[800:1000]
    data['testX']  = X[1000:]
    data['trainY'] = Y[:800]
    data['valY']   = Y[800:1000]
    data['testY']  = Y[1000:]
    return data


############################################
# Sensor Split
############################################

def split_into_quadrants(X):
    '''
    Split flattened 28×28 grayscale images into 4 non-overlapping quadrants.

    Each quadrant corresponds to one sensor in the decentralized setting.

    Args:
        X (ndarray): Input array of shape (N, 784).

    Returns:
        index_blocks (list): list of 4 index arrays (each quadrant’s feature indices)
        X_blocks (list): list of 4 views of X (one per quadrant)
    '''
    N, M = X.shape
    assert M == 784, "Expected flattened 28×28 grayscale images (M=784)."

    def q_indices(qr, qc):
        rows = range(qr * 14, (qr + 1) * 14)
        cols = range(qc * 14, (qc + 1) * 14)
        idx = []
        for r in rows:
            base = r * 28
            for c in cols:
                idx.append(base + c)
        return np.array(idx, dtype=int)

    I1 = q_indices(0, 0)
    I2 = q_indices(0, 1)
    I3 = q_indices(1, 0)
    I4 = q_indices(1, 1)

    index_blocks = [I1, I2, I3, I4]
    X_blocks = [X[:, I] for I in index_blocks]
    return index_blocks, X_blocks


############################################
# Visualization (optional)
############################################

def visualize_features(feat_to_keep):
    '''
    Optional visualization of selected features (e.g., in feature selection).

    Args:
        feat_to_keep (list or ndarray): indices of selected features
    '''
    feat_len = len(feat_to_keep)
    feat_mask = np.zeros(784)
    feat_mask[feat_to_keep] = 1
    plt.imshow(feat_mask.reshape(28, 28), cmap='gray')
    plt.title(f'Selected {feat_len} Features')
    plt.show()


def visualize_2d_decision_boundary(model, X, Y, h=0.05):
    '''
    Optional 2-D visualization for Task 1 (synthetic dataset).

    Given a trained classifier (with predict() method), plots the decision regions.

    Args:
        model: classifier object with predict() method
        X (ndarray): 2D input data
        Y (ndarray): class labels
        h (float): grid resolution
    '''
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    try:
        Z = model.predict(grid)
    except Exception:
        Z = np.zeros(len(grid))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor='k', s=30)
    plt.title("Decision Regions (Task 1 optional visualization)")
    plt.show()


############################################
# Plotting (Task-Specific)
############################################

def plot_result_per_sensor(result3_1=None, ylim_lo=0.5):
    '''
    Task 3.1 — Decentralized with fixed per-sensor budget.

    Args:
        result3_1 (dict): expects keys:
            'k' — list of per-sensor budgets (ints)
            'test_accuracy' — list of floats
            'baseline_test_accuracy_task1' — float (optional)
    '''
    if result3_1 is None:
        return
    plt.plot(result3_1['k'], result3_1['test_accuracy'], label='Decentralized', marker='o', markersize=8)
    if 'baseline_test_accuracy_task1' in result3_1:
        plt.axhline(y=result3_1['baseline_test_accuracy_task1'], linestyle='--', color='gray', label='Task 1 baseline')
    plt.ylim(ylim_lo, 1)
    plt.legend()
    plt.xlabel("Per-Sensor Budget k (bits/image)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Task 3.1 — Accuracy vs Per-Sensor Budget", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_result_total(result2=None, result3_2=None, ylim_lo=0.5, baseline_acc = None):
    '''
    Task 2 (Centralized) and Task 3.2 (Decentralized Total Budget).

    Args:
        result2 (dict): centralized results with keys 'B_tot' and 'test_accuracy'
        result3_2 (dict): decentralized results with keys 'B_tot' and 'test_accuracy'
    '''
    if result2 is not None:
        plt.plot(result2['B_tot'], result2['test_accuracy'], label='Centralized', markersize=12, linewidth=3)
    if result3_2 is not None:
        plt.plot(result3_2['B_tot'], result3_2['test_accuracy'], label='Decentralized', marker='s', markersize=8)
    if baseline_acc is not None:
        plt.axhline(y=baseline_acc, linestyle='--', color='gray', label='Task 1 baseline')
    plt.ylim(ylim_lo, 1)
    plt.legend()
    plt.xlabel("Total Budget $B_{tot}$ (bits/image)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Accuracy vs Total Budget", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_result_centralized_vs_decentralized(result=None, ylim_lo=0.5):
    '''
    Comparison of centralized vs decentralized performance at matched budgets.

    Args:
        result (dict): expects keys:
            'centralized' — dict with 'B_tot' and 'test_accuracy'
            'decentralized' — dict with 'B_tot' and 'test_accuracy'
    '''
    if result is None:
        return
    cent = result.get('centralized', {})
    decz = result.get('decentralized', {})
    if cent:
        plt.plot(cent['B_tot'], cent['test_accuracy'], label='Centralized', marker='o', markersize=12, linewidth=3)
    if decz:
        plt.plot(decz['B_tot'], decz['test_accuracy'], label='Decentralized', marker='s', markersize=8)
    plt.ylim(ylim_lo, 1)
    plt.legend()
    plt.xlabel("Total Budget $B_{tot}$ (bits/image)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Centralized vs Decentralized Performance", fontsize=14)
    plt.grid(True)
    plt.show()


def plot_result_target(result=None):
    '''
    Task 3.3 — Minimal Bits for Target Accuracy α.

    Args:
        result (dict): expects keys:
            'alphas' — list of α values
            'centralized_min_bits' — list of ints (optional)
            'decentralized_min_bits' — list of ints (optional)
    '''
    if result is None:
        return
    if 'centralized_min_bits' in result:
        plt.plot(result['alphas'], result['centralized_min_bits'], label='Centralized', marker='o', markersize=12, linewidth=3)
    if 'decentralized_min_bits' in result:
        plt.plot(result['alphas'], result['decentralized_min_bits'], label='Decentralized', marker='s', markersize=8)
    plt.legend()
    plt.xlabel("Target Accuracy α", fontsize=12)
    plt.ylabel("Minimal Total Bits", fontsize=12)
    plt.title("Task 3.3 — Minimal Bits vs Target Accuracy", fontsize=14)
    plt.grid(True)
    plt.show()



"""
Saved code
from utils import (
    prepare_mnist_data,
    split_into_quadrants,
    plot_result_per_sensor,
)

# -----------------------------------
# Load Fashion-MNIST (Task 3 uses MNIST only)
# -----------------------------------
mnist_data = prepare_mnist_data()
trainX, trainY = mnist_data["trainX"], mnist_data["trainY"]
valX,   valY   = mnist_data["valX"],   mnist_data["valY"]
testX,  testY  = mnist_data["testX"],  mnist_data["testY"]

# -----------------------------------
# Task 1 baseline (unquantized classifier accuracy)
# -----------------------------------
base_clf = MyDecentralized(K=3, lam=1e-3, solver="HIGHS", verbose=False)
base_clf.train(trainX, trainY)
baseline_acc = base_clf.evaluate(testX, testY)
print(f"[Task 1] baseline test accuracy (unquantized) = {baseline_acc:.4f}")

# -----------------------------------
# Split images into quadrants (one per sensor)
# -----------------------------------
_, train_blocks = split_into_quadrants(trainX)
_, val_blocks   = split_into_quadrants(valX)
_, test_blocks  = split_into_quadrants(testX)

# Each block has d_s features (should be 196)
_, d_s = train_blocks[0].shape

# -----------------------------------
# Task 3.1: decentralized, fixed per-sensor budget k
# -----------------------------------
fc = MyFeatureCompression(K=3, lam=1e-2, solver="HIGHS", verbose=False)

# Choose per-sensor budgets k; make them multiples of d_s so that
# k = b_s * d_s for nice integer bit-depths b_s.
# e.g. b_s in {1, 2, 3, 4} → k in {196, 392, 588, 784}
bs_values = [1, 2, 3, 4, 5, 6]
k_values = [b_s * d_s for b_s in bs_values]

test_accs_3_1 = []

for k in k_values:
    acc_dec, B_tot = fc.train_and_evaluate_decentralized_fixed_k(
        train_blocks, val_blocks, test_blocks,
        trainY, valY, testY,
        k=k
    )
    test_accs_3_1.append(acc_dec)
    print(f"[Task 3.1] k={k} bits/image/sensor (b_s={k // d_s}), "
        f"B_tot={B_tot}, test_acc={acc_dec:.4f}")

# Package results in the format expected by utils.plot_result_per_sensor
result3_1 = {
    "k": k_values,                          # per-sensor budgets on x-axis
    "test_accuracy": test_accs_3_1,         # decentralized accuracy curve
    "baseline_test_accuracy_task1": baseline_acc,
}

# Plot: Accuracy vs per-sensor budget k
plot_result_per_sensor(result3_1=result3_1, ylim_lo=0.5)


"""