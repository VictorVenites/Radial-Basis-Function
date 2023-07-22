"""
Classe e métodos para implementar a Regressão por multiplicação de Matriz Pseudo-Inversa.

Original Author: Victor Venites
"""

import numpy as np
import pandas as pd
import time

class PseudoInversa:
    """Multiplicacao de Matriz Pseudo-Inversa
    Original Author: Victor Venites
    """
    def __init__(self, bias = 1):
        self.bias = bias
        self.weights = None
        self.feature_names_in_ = None
        self.n_feature_in_ = None
        self.n_linhas_in_ = None
        self.time = None
    
    def bias_increment(self, dados):
        if self.bias != 0:
            if self.n_linhas_in_ > self.n_feature_in_:
                dados["Bias_Vies_Intercept"] = self.bias
        return dados

    def fit(self, X, y):
        try:
            self.feature_names_in_ = X.columns
            self.n_feature_in_ = X.shape[1]
            self.n_linhas_in_ = X.shape[0]
        except:
            self.n_feature_in_ = len(X[0])
            self.n_linhas_in_ = len(X)
        Matriz_X = pd.DataFrame(X)
        Matriz_X = self.bias_increment(Matriz_X)
        tempo_Inicial = time.time()
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
        self.time = time.time() - tempo_Inicial

    def predict(self, X):
        Matriz_X = pd.DataFrame(X)
        Matriz_X = self.bias_increment(Matriz_X)
        return np.dot(Matriz_X, self.weights)

    def score(self, x, y):
        # TODO: adicionar métricas do próprio Sklearn
        pass
