import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pydot
from IPython.display import Image

# Carregar o dataset California Housing
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target

# Dividir o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo de árvore de regressão
tree_reg = DecisionTreeRegressor(max_depth=3)
tree_reg.fit(X_train, y_train)

# Exportar a árvore de decisão para um arquivo .dot
export_graphviz(
    tree_reg,
    out_file="img/california_tree.dot",
    feature_names=california_housing.feature_names,
    rounded=True,
    filled=True
)

# Gerar uma imagem da árvore de decisão
(graph,) = pydot.graph_from_dot_file("img/california_tree.dot")
graph.write_png("img/california_tree.png")

# Mostrar a imagem da árvore de decisão
Image("img/california_tree.png")

# Avaliar o modelo no conjunto de teste
y_pred = tree_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro quadrático médio (MSE) no conjunto de teste: {mse:.2f}")

# Plotar valores reais vs. valores previstos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Valores Previstos (Árvore de Regressão)')
plt.show()
