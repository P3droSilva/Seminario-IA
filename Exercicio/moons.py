import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

# Step a: Generate moons dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)


# Step b: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step c: Use GridSearchCV to find good hyperparameters
param_grid = {'max_leaf_nodes': list(range(2, 100))}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Step d: Train on the full training set using the best parameters
model = DecisionTreeClassifier(**best_params, random_state=42)
model.fit(X_train, y_train)

export_graphviz(
    model,
    out_file="img/moons_tree.dot",
    feature_names =["feature1", "feature2"],  # Custom feature names for the moons dataset
    class_names = ["class0", "class1"],
    rounded=True,
    filled=True
)

(graph,) = pydot.graph_from_dot_file("img/moons_tree.dot")
graph.write_png("img/moons_tree.png")

Image("img/moons_tree.png")

# Measure performance on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Best hyperparameters: {best_params}")
