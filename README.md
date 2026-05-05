# Decentralized Classification Under Feature Compression

This project studies how feature compression affects multiclass image classification in both centralized and decentralized sensing settings. Using a linear programming based classifier, we compare full precision classification against quantized feature representations under different bit budget constraints.

## Objective

To study how feature compression and decentralized sensing affect multiclass image classification accuracy by comparing a full precision LP based classifier against centralized and per sensor quantized versions under varying bit budgets.

## Project Overview

The project is divided into three main tasks:

1. **Build a Classifier**
   - Implement a multiclass linear classifier using the Crammer Singer margin formulation.
   - Use slack variables and L1 regularization to formulate the classifier as a linear program.
   - Evaluate the classifier on a synthetic dataset and a reduced three class Fashion MNIST dataset.

2. **Build a Quantizer**
   - Apply uniform scalar quantization to image features.
   - Study the relationship between total bit budget and classification accuracy.
   - Compare compressed performance against the full precision baseline.

3. **Feature Compression in a Decentralized Setting**
   - Split each Fashion MNIST image into four non overlapping quadrants.
   - Treat each quadrant as a separate sensor with its own local feature block.
   - Evaluate classification accuracy under fixed per sensor budgets and fixed total budgets.
   - Explore whether balanced or uneven bit allocations perform better.

## Methods

The classifier is formulated as a linear program using a multiclass hinge loss with margin constraints. The model uses L1 regularization on the weights and bias terms to promote sparsity. Features are standardized using training set statistics before classification.

For compression, the project uses a simple uniform scalar quantizer. Each pixel is normalized using training set minimum and maximum values, mapped to a finite number of quantization levels, and then reconstructed before being passed into the classifier.

In the decentralized setting, the image is divided into four feature blocks, each corresponding to a sensor. Each sensor independently quantizes its local features before the reconstructed feature blocks are concatenated and passed to the classifier.

## Datasets

- Synthetic 2D dataset
- Reduced Fashion MNIST dataset
  - 3 classes
  - Flattened 28 × 28 grayscale images
  - 784 total features per image

## Key Results

The full precision LP classifier achieved strong baseline performance, with high accuracy on both the synthetic dataset and the reduced Fashion MNIST dataset. Under centralized quantization, accuracy improved as the bit budget increased, with roughly 3 bits per pixel preserving most of the classifier’s performance. In the decentralized setting, splitting the image across four sensors did not significantly reduce accuracy when the total bit budget was comparable. Balanced sensor allocations generally performed best, suggesting that all four image quadrants contributed similarly useful information.

## Tech Stack

- Python
- NumPy
- CVXPY or linear programming solver
- Matplotlib
- Fashion MNIST dataset
- Linear programming based optimization
- Uniform scalar quantization

## Main Takeaways

This project shows that simple uniform quantization can preserve most of the useful information needed for classification, even under fairly aggressive compression. It also suggests that decentralized sensing does not necessarily harm classification accuracy when each sensor receives a reasonable bit budget and the classifier is trained on the resulting compressed features.
