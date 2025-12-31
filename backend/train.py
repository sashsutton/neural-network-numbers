import numpy as np
import tensorflow as tf

# Load MNIST Data
print("Downloading MNIST data...")
(X_train_raw, y_train_raw), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess and Extend Dataset
X_digits = X_train_raw.reshape(-1, 784) / 255.0
y_digits_labels = y_train_raw

# Generate "Not a Number" training samples
print("Generating 'Not a Number' training samples...")
num_nan_samples = 10000
X_nan = np.random.rand(num_nan_samples, 784) * 0.4
X_nan_inverted = 1.0 - (X_digits[:num_nan_samples])
X_train = np.vstack([X_digits, X_nan, X_nan_inverted])

def create_labels(y_digits, total_samples):
    oh = np.zeros((total_samples, 11))
    for idx, val in enumerate(y_digits):
        oh[idx, val] = 1
    oh[len(y_digits):, 10] = 1
    return oh

y_train = create_labels(y_digits_labels, len(X_train))


L0, L1, L2, L3, L4 = 784, 512, 256, 128, 11

# He Initialisation for 4 layers
W1 = np.random.randn(L1, L0) * np.sqrt(2. / L0)
b1 = np.zeros((L1, 1))
W2 = np.random.randn(L2, L1) * np.sqrt(2. / L1)
b2 = np.zeros((L2, 1))
W3 = np.random.randn(L3, L2) * np.sqrt(2. / L2)
b3 = np.zeros((L3, 1))
W4 = np.random.randn(L4, L3) * np.sqrt(2. / L3)
b4 = np.zeros((L4, 1))

# Activation Functions
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum(axis=0)

# Hyperparameters
batch_size = 32
learning_rate = 0.003 # Lowered slightly for deeper architecture stability
epochs = 25
lambda_reg = 0.001

print(f"Starting Training: {epochs} epochs...")

for epoch in range(epochs):
    permutation = np.random.permutation(len(X_train))
    X_train, y_train = X_train[permutation], y_train[permutation]

    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i+batch_size].T
        y_batch = y_train[i:i+batch_size].T
        m = x_batch.shape[1]

        # Forward Pass
        a1 = relu(np.dot(W1, x_batch) + b1)
        a2 = relu(np.dot(W2, a1) + b2)
        a3 = relu(np.dot(W3, a2) + b3)
        a4 = softmax(np.dot(W4, a3) + b4)

        # Backpropagation
        dz4 = a4 - y_batch
        dW4 = (np.dot(dz4, a3.T) / m) + (lambda_reg * W4)
        db4 = np.sum(dz4, axis=1, keepdims=True) / m

        dz3 = np.dot(W4.T, dz4) * relu_deriv(np.dot(W3, a2) + b3)
        dW3 = (np.dot(dz3, a2.T) / m) + (lambda_reg * W3)
        db3 = np.sum(dz3, axis=1, keepdims=True) / m

        dz2 = np.dot(W3.T, dz3) * relu_deriv(np.dot(W2, a1) + b2)
        dW2 = (np.dot(dz2, a1.T) / m) + (lambda_reg * W2)
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        dz1 = np.dot(W2.T, dz2) * relu_deriv(np.dot(W1, x_batch) + b1)
        dW1 = (np.dot(dz1, x_batch.T) / m) + (lambda_reg * W1)
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # Weight Updates
        W4 -= learning_rate * dW4
        b4 -= learning_rate * db4
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch + 1}/{epochs} complete")

# Save all weights including the new layer
np.savez("weights.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)