import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Sample data: features (age, income, student status) and labels (buy_computer)
X = np.array([[25, 45000, 0], [30, 58000, 1], [35, 75000, 1], [22, 38000, 0], [40, 96000, 1], [28, 67000, 0], [18, 20000, 0], [35, 80000, 1], [45, 105000, 1], [25, 50000, 0]])
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])  # 0: will not buy, 1: will buy

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
