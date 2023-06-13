"""
Classe e métodos para implementar a Regressão por multiplicação de Matriz Pseudo-Inversa.

Original Author: Victor Venites
"""

import numpy as np
import pandas as pd

class PseudoInversa:
    """Multiplicacao de Matriz Pseudo-Inversa
    Original Author: Victor Venites
    """
    def __init__(self, bias = 1):
        self.weights = None
        self.bias = bias
        self.feature_names_in_ = None
        self.n_feature_in_ = None

    def fit(self, X, y):
        try:
            self.feature_names_in_ = X.columns
            self.n_feature_in_ = X.shape[1]
        except:
            self.n_feature_in_ = len(X[0])
        Matriz_X = pd.DataFrame(X)
        if self.bias != 0:
            Matriz_X["Bias_Vies_Intercept"] = self.bias
        # Base_T * Base
        Matriz_Quadrada = np.dot(Matriz_X.T, Matriz_X)
        # Matriz Inversa
        try:
            Matriz_Inversa = np.linalg.inv(Matriz_Quadrada)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                Matriz_Regulada = Matriz_Quadrada + np.identity(len(Matriz_Quadrada))
                Matriz_Inversa = np.linalg.inv(Matriz_Regulada)
        # Inversa * Base_Dados_T
        Matriz_Pseudo_Inversa = pd.DataFrame(np.dot(Matriz_Inversa, Matriz_X.T))
        # Isola o peso final
        self.weights = np.dot(Matriz_Pseudo_Inversa, y)

    def predict(self, X):
        Matriz_X = pd.DataFrame(X)
        if self.bias != 0:
            Matriz_X["Bias_Vies_Intercept"] = self.bias
        return np.dot(Matriz_X, self.weights)

    def score(self, x, y):
        pass
