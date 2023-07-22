# Radial Basis Function: Regressão e Classificação facilitados por Matrizes

RBF propõe uma alternativa de cálculo matricial aos seus modelos de Regressão, Séries Temporais, Classificação e Convolução.

## Instalação

Para instalar a biblioteca na sua máquina, pelo prompt ou shell.

```

pip install radial-basis-function

```

Para usar a dependencia no código python.

```

import radial_basis_function

```

## Matriz Pseudo-Inversa

Multiplicação de Matriz Pseudo-Inversa, que é operado como alternativa no lugar da otimização por derivadas parciais. Refere-se a uma abordagem matemática chamada de "fórmula de Moore-Penrose", onde a matriz pseudo-inversa é utilizada para encontrar os coeficientes ótimos da equação de regressão. É uma generalização da matriz inversa para casos em que a matriz não é inversível (ou seja, singular), e a fórmula fornece uma maneira de obter uma solução única mesmo quando a matriz não é invertível.

Para usar a dependencia no código python, foram criados os métodos fit e predict com nome inspirados nos moldes do Sklearn.

```

# Instancia do modelo
modelo_pseudo_inversa = radial_basis_function.PseudoInversa()

# Treinando o modelo
modelo_pseudo_inversa.fit(Matriz_X, y)

# Previsão do modelo
predicao = modelo_pseudo_inversa.predict(Matriz_X)

# Para métrica de previsão utilize uma biblioteca já existente como sklearn.metrics
from sklearn import metrics

# Como:
metrics.mean_absolute_error(Y_Treino, predicao)
metrics.r2_score(Y_Treino, predicao)
metrics.accuracy_score(Y_Treino, predicao)
metrics.balanced_accuracy_score(Y_Treino, predicao)
metrics.f1_score(Y_Treino, predicao)

```

## Radial Basis Function

A estrutura básica de um Radial Basis Function (RBF Network) consiste em três componentes principais: as funções de base radial, os pesos associados a essas funções e uma camada de saída linear solucionada pela matriz pseudo-inversa.

Para usar a dependencia no código python, foram criados os métodos fit e predict com nome inspirados nos moldes do Sklearn.

```

# Instancia do modelo
modelo_rbf = radial_basis_function.RadialBasisFunction()

# Treinando o modelo
modelo_rbf.fit(Matriz_X, y)

# Previsão do modelo
predicao = modelo_rbf.predict(Matriz_X)

# Para métrica de previsão utilize uma biblioteca já existente como sklearn.metrics
from sklearn import metrics

# Como:
metrics.mean_absolute_error(Y_Treino, predicao)
metrics.r2_score(Y_Treino, predicao)
metrics.accuracy_score(Y_Treino, predicao)
metrics.balanced_accuracy_score(Y_Treino, predicao)
metrics.f1_score(Y_Treino, predicao)

```

## Testes

Teste sendo desenvolvido para Jupyter Notebook

by Victor Venites
