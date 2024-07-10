from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydot
from IPython.display import Image

# Carregar o dataset "wine"
wine = load_wine()
X = wine.data
y = wine.target

# Dividir o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Criar e treinar a árvore de decisão
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

print("Árvore de decisão treinada com sucesso!")

# Exportar a árvore de decisão para um arquivo .dot
export_graphviz(
    tree_clf,
    out_file="img/wine_tree.dot",
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    rounded=True,
    filled=True
)

# Gerar uma imagem da árvore de decisão
(graph,) = pydot.graph_from_dot_file("img/wine_tree.dot")
graph.write_png("img/wine_tree.png")

# Mostrar a imagem da árvore de decisão
Image("img/wine_tree.png")
print("Árvore de decisão gerada com sucesso!")

# Classificar os dados de teste
y_pred = tree_clf.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"A precisão do modelo na classificação dos dados de teste é: {accuracy:.2f}")

