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
iris_df['species'] = iris.target_names[iris.target]

# First part of the code (petal width as feature)
# Select the petal width (feature) and target (1 for Iris Virginica, 0 for others)
X1 = iris_df[['petal width (cm)']].values
y1 = (iris_df['target'] == 2).astype(int)  # 1 for Iris Virginica, 0 for others

# Standardize the feature (important for logistic regression)
scaler1 = StandardScaler()
X1 = scaler1.fit_transform(X1)

# Split data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Create a logistic regression model
model1 = LogisticRegression()

# Train the model
model1.fit(X1_train, y1_train)

# Predict on the test set
y1_pred = model1.predict(X1_test)

# Evaluate the model
accuracy1 = accuracy_score(y1_test, y1_pred)
conf_matrix1 = confusion_matrix(y1_test, y1_pred)
class_report1 = classification_report(y1_test, y1_pred)

print("First Logistic Regression Model Metrics:")
print(f'Accuracy: {accuracy1:.2f}')
print(f'Confusion Matrix:\n{conf_matrix1}')
print(f'Classification Report:\n{class_report1}')

# Second part of the code (petal length and width as features)
# Select only petal length and width as features, and filter for Iris Virginica
X2 = iris_df[['petal length (cm)', 'petal width (cm)']].values
y2 = (iris_df['species'] == 'virginica').astype(int)  # 1 for Iris Virginica, 0 for others

# Standardize the features (important for logistic regression)
scaler2 = StandardScaler()
X2 = scaler2.fit_transform(X2)

# Split data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Create a logistic regression model
model2 = LogisticRegression()

# Train the model
model2.fit(X2_train, y2_train)

# Define the decision boundary plot
plt.figure(figsize=(18, 6))

# Plot decision boundary for first model (petal width)
plt.subplot(1, 2, 1)
X1_min, X1_max = X1[:, 0].min() - 1, X1[:, 0].max() + 1
X1_plot = np.linspace(X1_min, X1_max, 1000).reshape(-1, 1)
probabilities1 = model1.predict_proba(X1_plot)
plt.plot(X1_plot, probabilities1[:, 1], color='blue', label='Iris Virginica')
plt.plot(X1_plot, probabilities1[:, 0], color='green', label='Not Iris Virginica')
plt.scatter(X1_test, y1_test, c=y1_test, cmap='coolwarm', edgecolors='k', s=100, label='Test data (Petal Width)')
plt.title('Logistic Regression Decision Boundary (Petal Width)')
plt.xlabel('Petal Width (standardized)')
plt.ylabel('Probability')
plt.legend()
plt.xlim(X1_min, X1_max)
plt.ylim(-0.1, 1.1)

# Plot decision boundary for second model (petal length and width)
plt.subplot(1, 2, 2)
# Generate grid of feature values
x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Predict probabilities for each point in the grid
Z = model2.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot contour of probabilities
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

# Scatter plot of training set
plt.scatter(X2_train[:, 0], X2_train[:, 1], c=y2_train, cmap=plt.cm.RdBu, edgecolors='k', s=30)
plt.xlabel('Petal length (standardized)')
plt.ylabel('Petal width (standardized)')
plt.title('Logistic Regression Decision Boundary and Probability Contours (Petal Length & Width)')
plt.colorbar(label='Probability')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Plot decision boundary (0.5 probability)
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k', linestyles='dashed')

# Plot probability levels
plt.contour(xx, yy, Z, levels=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9], linewidths=1, colors='b')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
