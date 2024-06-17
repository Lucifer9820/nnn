import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons

# Load the iris dataset
iris = datasets.load_iris()
X_iris = iris["data"][:, (2, 3)]  # Petal length, petal width
y_iris = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica has code 2

# Train a linear SVM model
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42))
])
svm_clf.fit(X_iris, y_iris)

# Predict for a sample iris flower with petal length 5.5 and petal width 1.7
sample_prediction = svm_clf.predict([[5.5, 1.7]])  # Detected as Iris Virginica
print("Prediction for sample [5.5, 1.7]:", sample_prediction)

# Plot decision boundary for linear SVM on Iris dataset
def plot_iris_decision_boundary(model, X, y):
    x0s = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1s = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = model.predict(X_new).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, alpha=0.2, cmap=plt.cm.brg)
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='green', marker='^')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='blue', marker='s')
    plt.xlabel("Petal length")
    plt.ylabel("Petal width")
    plt.title("Decision Boundary for Linear SVM on Iris Dataset")

plt.figure(figsize=(8, 6))
plot_iris_decision_boundary(svm_clf, X_iris, y_iris)
plt.show()

# Plotting Moons data to illustrate its linear inseparability
X_moons, y_moons = make_moons(n_samples=100, noise=0.4)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")  # bs stands for blue square
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")  # g^ stands for green triangle
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plt.figure(figsize=(8, 6))
plot_dataset(X_moons, y_moons, [-1.5, 2.5, -1, 1.5])
plt.title("Moons Dataset")
plt.show()

# Train a non-linear SVM model with polynomial features
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss='hinge', random_state=42))
])
polynomial_svm_clf.fit(X_moons, y_moons)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

# Plot predictions and decision boundary for polynomial SVM on Moons dataset
plt.figure(figsize=(8, 6))
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X_moons, y_moons, [-1.5, 2.5, -1, 1.5])
plt.title("Polynomial SVM on Moons Dataset")
plt.show()
