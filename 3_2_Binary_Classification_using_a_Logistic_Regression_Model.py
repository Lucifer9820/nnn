import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = load_iris()

# Create a DataFrame for easier manipulation
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris.target_names[iris.target]

# Select only petal length and width as features, and filter for Iris Virginica
X = iris_df[['petal length (cm)', 'petal width (cm)']].values
y = (iris_df['species'] == 'virginica').astype(int)  # 1 for Iris Virginica, 0 for others

# Standardize the features (important for logistic regression)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Define the decision boundary plot
plt.figure(figsize=(8, 6))

# Plot decision boundary (dashed line at 0.5 probability)
# Generate grid of feature values
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict probabilities for each point in the grid
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

# Plot contour of probabilities
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)

# Scatter plot of training set
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k', s=30)
plt.xlabel('Petal length (standardized)')
plt.ylabel('Petal width (standardized)')
plt.title('Logistic Regression Decision Boundary and Probability Contours')
plt.colorbar(label='Probability')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

# Plot decision boundary (0.5 probability)
plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k', linestyles='dashed')

# Plot probability levels
plt.contour(xx, yy, Z, levels=[0.15, 0.3, 0.45, 0.6, 0.75, 0.9], linewidths=1, colors='b')

plt.show()
