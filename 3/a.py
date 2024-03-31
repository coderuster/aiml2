import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.randint(2, size=100)  # Binary classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network model
class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.rand(2)
        self.bias = np.random.rand(1)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def train(self, X, y, epochs=100, learning_rate=0.1):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Backward pass (gradient descent)
            error = y - y_pred
            self.weights += learning_rate * np.dot(X.T, error)
            self.bias += learning_rate * np.sum(error)

# Train the neural network model
model = NeuralNetwork()
model.train(X_train, y_train)

# Make predictions on the test data
y_pred = np.round(model.predict(X_test))

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
