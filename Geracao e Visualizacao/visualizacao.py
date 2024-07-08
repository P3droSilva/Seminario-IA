from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot
from IPython.display import Image

iris = load_iris()
X = iris.data[:, 2:]  # comprimento e largura das pétalas
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

print("Árvore de decisão treinada com sucesso!")

export_graphviz(
    tree_clf,
    out_file="img/iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

(graph,) = pydot.graph_from_dot_file("img/iris_tree.dot")
graph.write_png("img/iris_tree.png")

Image("img/iris_tree.png")
print("Árvore de decisão gerada com sucesso!")
