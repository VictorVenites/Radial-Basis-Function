"""
Classe e métodos para implementar a Regressão por Radial Basis Function

Original Author: Victor Venites
"""

# Dependencias internas á Biblioteca
from .pseudo_inversa import PseudoInversa
# Dependencias Externas
import numpy as np
import pandas as pd
import time


class RadialBasisFunction:
    """Radial Basis Function
    Original Author: Victor Venites
    """
    def __init__(self, funcao = 'Gaussiana',
                 Qtd_Polos = 0, Polos_iniciais_fixos = False, Polos_Otimizados = None):
        self.funcao = funcao
        self.Qtd_Polos = Qtd_Polos
        self.Polos_iniciais_fixos = Polos_iniciais_fixos
        self.C = Polos_Otimizados
        self.sigma = None
        self.pesos = None
        self.R = None
        self.feature_names_in_ = None
        self.n_feature_in_ = None
        self.n_linhas_in_ = None
        self.time = None
        
    # # # Funções Base
    def FuncoesBase(self, t, n_PoloAtual, N_TotalPolos, X, C):
        # TODO: revisar as funções, para se aproximarem do original de inspiração
        Gama = 1 / (2 * self.sigma ** 2) # Por Convensão
        Radial = np.linalg.norm(X - C)
        #Radial = (np.linalg.norm(X - C)) ** 2 # 
        if self.funcao == "Gaussiana":
            calculo = np.exp(-Gama * (Radial ** 2))
            #calculo = np.exp(- Gama * Radial)
        #elif funcao == "Multiquadratica":
        #    calculo = np.sqrt(Radial + (1 / Gama) ** 2)
        elif self.funcao == "Sigmoide":
            calculo = np.tanh(-Gama * (Radial ** 2))
        elif self.funcao == "Senoidal":
            calculo = np.sin(-Gama * (Radial ** 2))
        elif self.funcao == "Logistica":
            calculo = 1 / (1 + np.exp(Gama * (Radial**2)))
        elif self.funcao == "ReLu":
            calculo = max(0, Radial)
        elif self.funcao == "Fractal":
            # Seed Function 1 /((x**2 + y**2 + 1)**(1/2))
            calculo = 1 / ((Radial**2 + Gama**2 + 1) ** (1/2))
        elif self.funcao == "Aurea":
            golden = (1 + 5 ** 0.5) / 2
            calculo = golden * Gama * Radial
        elif self.funcao == "Fourier":
            # Altura_da_Onda * Sinal( CICLO * TEMPO_X / Tamanho_Periodo )
            calculo = np.sqrt(2) * np.sin(- Gama * 2 * np.pi * t * (Radial ** 2) * n_PoloAtual)
        return calculo

    # # # Radial - Cálculo
    def Radial(self, X):
        variaveis = pd.DataFrame(X)
        num_de_licoes = variaveis.shape[0]
        num_de_variaveis = variaveis.shape[1]
        if self.Qtd_Polos == "50percent":
            if num_de_variaveis > 5:
                self.Qtd_Polos = int(round(num_de_variaveis/2))

        # Coordenada dos Polos:
        if num_de_variaveis <= 2:
            num_de_polos = 2
        elif num_de_variaveis > 2:
            num_de_polos = num_de_variaveis
        if self.Qtd_Polos > 1:
            num_de_polos = self.Qtd_Polos
        if self.Qtd_Polos > num_de_licoes:
            num_de_polos = num_de_licoes - 2
        if self.Qtd_Polos > num_de_variaveis * 8:
            # TODO: Avaliar uma bordagem melhor para evitar Overfitting
            num_de_polos = int(round((self.Qtd_Polos + num_de_variaveis * 8) / 9 + 2))
        if num_de_polos > num_de_licoes:
            #num_de_polos = num_de_licoes - 2
            num_de_polos = num_de_licoes

        # Gerando os Polos Aleatórios
        # TODO: Centróides do K-means, ou gerados por Algoritmo Genéticos)
        C = np.zeros((num_de_polos, num_de_variaveis), dtype = float)
        C = np.random.rand(num_de_polos, num_de_variaveis)
        # Limite por Coluna (apenas entre as escalas de cada coluna)
        Limite = np.array(variaveis.max() - variaveis.min())
        C = pd.DataFrame(C) * Limite + np.array(variaveis.min())
        C = np.array(C)
        
        # TODO: Criar Função específica para geração dos Polos
        #if self.C is not None:

        dist_entre_os_polos = np.zeros((num_de_polos, num_de_polos))
        for i in range(0, num_de_polos, 1):
            for j in range(0, num_de_polos, 1):
                dist_entre_os_polos[i, j] = np.linalg.norm(C[i] - C[j])
        if self.Polos_iniciais_fixos:
            C[0] = variaveis.mean()
            C[1] = variaveis.std()
            if num_de_polos > 2:
                C[2] = variaveis.max()
            if num_de_polos > 3:
                C[3] = variaveis.min()
        dps_max = np.max(dist_entre_os_polos)
        self.sigma = dps_max / np.sqrt(2 * num_de_polos)
        variaveis = np.array(variaveis)

        # Matrix [R]:
        # R = Matrix de Base Radial [R]
        R = np.zeros((num_de_licoes, num_de_polos))
        for n in range(0, num_de_licoes, 1):
            # Input Layer
            for i in range(0, num_de_polos, 1):
                # Hidden Layer
                R[n, i] = self.FuncoesBase(n, (i+1), len(C), variaveis[n], C[i])
        #print(f"R.shape -> {R.shape}")
        R[R.shape[1]] = 1
        #print(f"R.shape -> {R.shape}")
        self.R = pd.DataFrame(R)
        self.C = C

    def fit(self, X, Matriz_Y):
        try:
            self.feature_names_in_ = X.columns
            self.n_feature_in_ = X.shape[1]
            self.n_linhas_in_ = X.shape[0]
        except:
            self.n_feature_in_ = len(X[0])
            self.n_linhas_in_ = len(X)
        tempo_Inicial = time.time()
        self.Radial(X)
        Matriz_Pseudo_Inversa = PseudoInversa()
        Matriz_Pseudo_Inversa.fit(np.array(self.R), Matriz_Y)
        self.pesos = Matriz_Pseudo_Inversa.weights
        self.time = time.time() - tempo_Inicial

    def predict(self, X):
        #A = np.dot(variaveis_X, W) # Valores finais da predição
        variaveis_X = np.array(X)
        Predicao_Y = np.zeros(len(variaveis_X))
        W = [i for i in self.pesos]
        C = self.C
        
        for i in range(0, len(variaveis_X), 1):
            # Input
            Predicao_Y[i] = W[-1]
            for j in range(0, len(W) - 1, 1):
                # Somatorio + Pesos * Funcoes_Ativa
                Predicao_Y[i] = Predicao_Y[i] + W[j] * self.FuncoesBase(i, (i+1), len(C), variaveis_X[i], C[j])

        return Predicao_Y

    def score(self, x, y):
        # TODO: adicionar métricas do próprio Sklearn
        pass
    