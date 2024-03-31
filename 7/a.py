import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Dense


# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = np.random.randint(2, size=100)  # Binary classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),  # 1st hidden layer with 64 neurons
    Dense(32, activation='relu'),  # 2nd hidden layer with 32 neurons
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
