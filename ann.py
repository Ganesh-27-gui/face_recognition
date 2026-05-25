import numpy as np

class ANN:
    """Simple feedforward neural network with backpropagation."""
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        # Xavier initialization
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_deriv(self, a):
        return a * (1 - a)

    def softmax(self, z):
        z -= np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        """X shape: (input_size, batch)"""
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, Y_pred, Y_true):
        """Cross-entropy loss. Y_true: one-hot (classes, batch)"""
        m = Y_true.shape[1]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-10)) / m
        return loss

    def backward(self, X, Y_true):
        m = X.shape[1]
        dZ2 = self.A2 - Y_true                          # (output, batch)
        dW2 = (dZ2 @ self.A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.sigmoid_deriv(self.A1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=500):
        """X: (features, samples), Y: one-hot (classes, samples)"""
        for epoch in range(epochs):
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y_pred, Y)
            self.backward(X, Y)
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    def predict(self, X):
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=0)

    def predict_proba(self, X):
        return self.forward(X)