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
    def __init__(self, funcao = 'Gaussiana', bias = 1,
                 Qtd_Polos = 0, Polos_iniciais_fixos = False, Polos_Otimizados = None):
        self.funcao = funcao
        self.bias = bias
        self.Qtd_Polos = Qtd_Polos
        self.Polos_iniciais_fixos = Polos_iniciais_fixos
        self.C = Polos_Otimizados
        self.numero_polos_definidos = None
        self.sigma = None
        self.pesos = None
        self.R = None
        self.feature_names_in_ = None
        self.n_feature_in_ = None
        self.n_linhas_in_ = None
        self.time = None
    
    def definir_numero_de_polos(self):
        if self.Qtd_Polos == "50percent":
            if self.n_feature_in_ > 5:
                aux_polos_desejados = int(round(self.n_feature_in_/2))
        else:
            aux_polos_desejados = int(self.Qtd_Polos)

        # Coordenada dos Polos:
        if self.n_feature_in_ <= 2:
            self.numero_polos_definidos = 2
        else:
            if self.n_feature_in_ > 2:
                self.numero_polos_definidos = int(self.n_feature_in_)
            if aux_polos_desejados > 1:
                self.numero_polos_definidos = aux_polos_desejados
            if aux_polos_desejados > self.n_feature_in_ * 8:
                # TODO: Avaliar uma bordagem melhor para evitar Overfitting
                self.numero_polos_definidos = int(round((aux_polos_desejados + self.n_feature_in_ * 8) / 9))
            if self.numero_polos_definidos > self.n_linhas_in_:
                # -1 para evitar Matriz Indeterminada após adicionar Bias
                self.numero_polos_definidos = self.n_linhas_in_ - 1
    
    def Gerar_Polos(self, variaveis):
        # Gerando os Polos Aleatórios
        # TODO: Centróides do K-means, ou gerados por Algoritmo Genéticos)
        if self.C is None:
            C = np.zeros((self.numero_polos_definidos, self.n_feature_in_), dtype = float)
            C = np.random.rand(self.numero_polos_definidos, self.n_feature_in_)
            # Limite por Coluna (apenas entre as escalas de cada coluna)
            Limite = np.array(variaveis.max() - variaveis.min())
            C = pd.DataFrame(C) * Limite + np.array(variaveis.min())
            C = np.array(C)
            dist_entre_os_polos = np.zeros((self.numero_polos_definidos, self.numero_polos_definidos))
            for i in range(0, self.numero_polos_definidos, 1):
                for j in range(0, self.numero_polos_definidos, 1):
                    dist_entre_os_polos[i, j] = np.linalg.norm(C[i] - C[j])
            if self.Polos_iniciais_fixos:
                C[0] = variaveis.mean()
                C[1] = variaveis.std()
                if self.numero_polos_definidos > 2:
                    C[2] = variaveis.max()
                if self.numero_polos_definidos > 3:
                    C[3] = variaveis.min()
            dps_max = np.max(dist_entre_os_polos)
            self.sigma = dps_max / np.sqrt(2 * self.numero_polos_definidos)
            self.C = C
    
    # # # Funções Base
    def FuncoesBase(self, X, C, t, n_PoloAtual, N_TotalPolos):
        # TODO: revisar as funções, para se aproximarem do original de inspiração
        # TODO: Adicionar Função Aleatória
        Gama = 1 / (2 * self.sigma ** 2) # Por Convensão
        Radial = np.linalg.norm(X - C)
        #Radial = (np.linalg.norm(X - C)) ** 2 # 
        if self.funcao == "Gaussiana":
            calculo = np.exp(-Gama * (Radial ** 2))
            #calculo = np.exp(- Gama * Radial)
        elif self.funcao == "Multiquadratica":
            calculo = np.sqrt(Radial + (1 / Gama) ** 2)
        elif self.funcao == "Sigmoide":
            calculo = np.tanh(-Gama * (Radial ** 2))
        elif self.funcao == "Senoidal":
            calculo = np.sin(-Gama * (Radial ** 2))
        elif self.funcao == "Logistica":
            calculo = 1 / (1 + np.exp(Gama * (Radial**2)))
        elif self.funcao == "ReLu":
            calculo = max(0, Gama * Radial)
        elif self.funcao == "ELU":
            alpha = 1
            x = Gama * Radial
            if x >= 0:
                calculo = x
            else:
                calculo = alpha * (np.exp(x) - 1)
        elif self.funcao == "GELU":
            # Gaussian Error Linear Unit
            # https://towardsai.net/p/l/gelu-gaussian-error-linear-unit-code-python-tf-torch
            # acurado -> 0.5*x*(tanh[((2/pi)**(1/2))*(x + 0.044715*(x**(3)))])
            # rapido -> x*sigma*(1.702*x)
            x = Gama * Radial
            #calculo = 0.5*x*(np.tanh[((2/np.pi)**(1/2))*(x + 0.044715*(x**(3)))])
            calculo = 0.5*x*(np.tanh(((2/np.pi)**(1/2))*(x + 0.044715*(x**(3)))))
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
        self.definir_numero_de_polos()
        self.Gerar_Polos(pd.DataFrame(X))
        
        # TODO: Otimizar hidden layer
        variaveis = np.array(X)
        # Matrix [R]:
        # R = Matrix de Base Radial [R]
        R = np.zeros((self.n_linhas_in_, self.numero_polos_definidos))
        for n in range(0, self.n_linhas_in_, 1):
            # Input Layer
            for i in range(0, self.numero_polos_definidos, 1):
                # Hidden Layer
                R[n, i] = self.FuncoesBase(variaveis[n], self.C[i], n, (i+1), len(self.C))
        R = pd.DataFrame(R)
        #print(f"R.shape -> {R.shape}")
        # TODO: Revisar e discutir o Bias
        #R[R.shape[1]] = 1
        #print(f"R.shape -> {R.shape}")
        self.R = R
        

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
        # TODO: Melhorar o desempenho e estrutura do predict
        #A = np.dot(variaveis_X, W) # Valores finais da predição
        variaveis_X = np.array(X)
        Predicao_Y = np.zeros(len(variaveis_X))
        W = [i for i in self.pesos]
        
        for i in range(0, len(variaveis_X), 1):
            # Input
            Predicao_Y[i] = W[-1]
            for j in range(0, len(W) - 1, 1):
                # Somatorio + Pesos * Funcoes_Ativa
                Predicao_Y[i] = Predicao_Y[i] + W[j] * self.FuncoesBase(variaveis_X[i], self.C[j], i, (j+1), len(self.C))

        return Predicao_Y

    def score(self, x, y):
        # TODO: adicionar métricas do próprio Sklearn
        pass
    