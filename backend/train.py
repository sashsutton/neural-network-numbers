import numpy as np
import tensorflow as tf

# Load MNIST Data
print("Downloading MNIST data...")
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

# Preprocess and Extend Dataset
# Normalize digit data
X_digits = X_train_raw.reshape(-1, 784) / 255.0
y_digits_labels = y_train_raw

# Generate "Not a Number" data (Class 10)
# We use a mix of random noise and inverted MNIST patterns to help the model distinguish shapes
print("Generating 'Not a Number' training samples...")
num_nan_samples = 6000
X_nan = np.random.rand(num_nan_samples, 784) * 0.5 # Random noise
X_nan_inverted = 1.0 - (X_digits[:num_nan_samples]) # Inverted digits
X_nan_combined = np.vstack([X_nan, X_nan_inverted])

# Combine digits and NaN data
X_train = np.vstack([X_digits, X_nan_combined])

# Create 11-class One-Hot Labels
def create_labels(y_digits, total_samples):
    oh = np.zeros((total_samples, 11))
    # Fill 0-9 for MNIST digits
    for idx, val in enumerate(y_digits):
        oh[idx, val] = 1
    # Fill class 10 (Not a Number) for the rest
    oh[len(y_digits):, 10] = 1
    return oh

y_train = create_labels(y_digits_labels, len(X_train))

# 4. Architecture (784 -> 128 -> 11)
input_size = 784
hidden_size = 128
output_size = 11

# He Initialization (Optimized for ReLU)
W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((output_size, 1))

# Activation Functions
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum(axis=0)

# Training Hyperparameters
learning_rate = 0.01  # Lowered for more stable convergence with ReLU
lambda_reg = 0.001    # L2 Regularization / Weight Decay
epochs = 15           # Increased epochs

print(f"Starting Training: {hidden_size} hidden neurons, ReLU activation, 11 classes...")

for epoch in range(epochs):
    # Shuffle entire extended dataset
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]

    for i in range(len(X_train)):
        # Forward Pass
        a0 = X_train[i].reshape(784, 1)
        z1 = np.dot(W1, a0) + b1
        a1 = relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = softmax(z2)

        # Backpropagation
        dz2 = a2 - y_train[i].reshape(output_size, 1)
        # Weight decay applied to W2
        dW2 = np.dot(dz2, a1.T) + (lambda_reg * W2)
        db2 = dz2

        dz1 = np.dot(W2.T, dz2) * relu_deriv(z1)
        # Weight decay applied to W1
        dW1 = np.dot(dz1, a0.T) + (lambda_reg * W1)
        db1 = dz1

        # Updates
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch + 1}/{epochs} complete")

# Save the Brain
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Weights saved to weights.npz with 11-class support!")