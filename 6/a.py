import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample data: features (weight in grams, color intensity) and labels (0 for orange, 1 for mango)
X = np.array([[150, 0.8], [200, 0.6], [100, 0.9], [180, 0.7], [120, 0.85], [160, 0.75]])
y = np.array([0, 1, 0, 1, 0, 1])

# Initialize SVM classifier
clf = SVC(kernel='linear')

# Train the classifier
clf.fit(X, y)

# Test data
X_test = np.array([[170, 0.72], [130, 0.88], [190, 0.68]])

# Predict labels for test data
y_pred = clf.predict(X_test)

# Print predicted labels
print("Predicted labels:", y_pred)

# Evaluate accuracy
# Note: In real-world scenarios, you would need labeled test data to calculate accuracy
# Here, we assume we know the true labels for demonstration purposes.
true_labels = np.array([1, 0, 1])
accuracy = accuracy_score(true_labels, y_pred)
print("Accuracy:", accuracy)
