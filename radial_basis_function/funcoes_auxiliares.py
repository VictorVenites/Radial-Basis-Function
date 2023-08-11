"""
Funções de Suporte aos modelos:
- Vericiação
- Métricas

Original Author: Victor Venites
"""
import numpy as np
import pandas as pd


def check_Matrix_X(dados_x) -> tuple:
    nome_colunas = None
    if isinstance(dados_x, pd.DataFrame):
        nome_colunas = dados_x.columns
        n_feature_in_ = dados_x.shape[1]
        n_linhas_in_ = dados_x.shape[0]
    elif isinstance(dados_x, (tuple, list, np.ndarray)):
        if len(dados_x) > 0:
            if isinstance(dados_x[0], (tuple, list, np.ndarray)):
                n_feature_in_ = len(dados_x[0])
                n_linhas_in_ = len(dados_x)
            else:
                n_feature_in_ = 1
                n_linhas_in_ = len(dados_x)
        else:
            n_feature_in_ = 0
            n_linhas_in_ = 0
    else:
        raise ValueError('Unsupported input type')
    return nome_colunas, n_feature_in_, n_linhas_in_
