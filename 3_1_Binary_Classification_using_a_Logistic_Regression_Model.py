import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the iris dataset
iris = load_iris()

# Create a DataFrame for easier manipulation
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Select the petal width (feature) and target (1 for Iris Virginica, 0 for others)
X = iris_df[['petal width (cm)']].values
y = (iris_df['target'] == 2).astype(int)  # 1 for Iris Virginica, 0 for others

# Standardize the feature (important for logistic regression)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# Plot decision boundary
plt.figure(figsize=(8, 6))

# Plotting decision regions
X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
X_plot = np.linspace(X_min, X_max, 1000).reshape(-1, 1)
probabilities = model.predict_proba(X_plot)
plt.plot(X_plot, probabilities[:, 1], color='blue', label='Iris Virginica')
plt.plot(X_plot, probabilities[:, 0], color='green', label='Not Iris Virginica')
plt.scatter(X_test, y_test, c=y_test, cmap='coolwarm', edgecolors='k', s=100, label='Test data')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Petal Width (standardized)')
plt.ylabel('Probability')
plt.legend()
plt.xlim(X_min, X_max)
plt.ylim(-0.1, 1.1)
plt.show()
