import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate moon dataset
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train AdaBoost classifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42
)
ada_clf.fit(X_train, y_train)

# Train individual Decision Stump
ds_clf = DecisionTreeClassifier(max_depth=1, random_state=42)
ds_clf.fit(X_train, y_train)

# Evaluate accuracy of AdaBoost and Decision Stump on test data
y_pred_ada_clf = ada_clf.predict(X_test)
y_pred_ds_clf = ds_clf.predict(X_test)
print(ds_clf.__class__.__name__, accuracy_score(y_test, y_pred_ds_clf))
print(ada_clf.__class__.__name__, accuracy_score(y_test, y_pred_ada_clf))

# Plot decision boundary
def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

# Plot the decision boundary for AdaBoost classifier
plt.figure(figsize=(8, 6))
plot_decision_boundary(ada_clf, X, y)
plt.title("AdaBoost Classifier Decision Boundary")
plt.show()