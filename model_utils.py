"""Utilitários compartilhados pelos testes de vazamento de dados.

Configuração via variáveis de ambiente (todas opcionais):

    DATASETS      lista separada por vírgula, ex.: DATASETS=227_cpu_small,201_pol
    ALGS          lista separada por vírgula, ex.: ALGS=SVR,KNN
    NUM_TRIALS    número de repetições (default 6)
    N_JOBS        paralelismo do CV / GridSearchCV (default -1 = todos os núcleos)
    SVR_IMPL      "linear" (LinearSVR, rápido, default) ou "libsvm" (SVR(kernel="linear") original)
    SVR_SCALE_Y   "1" (default) padroniza o alvo dentro do SVR; "0" desliga
    SVR_MAX_ROWS  teto de linhas para SVR com kernel (libsvm / tuning). Default 5000; 0 = sem teto - utilizando sem teto
    SVR_CACHE_MB  cache do libsvm em MB (default 1000)
    RESUME        "1" pula blocos já gravados no CSV; "0" (default) recomeça do zero
"""
import os
import signal
import sys
import time
from contextlib import contextmanager


def signal_handler(sig, frame):
    print("Exiting gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.experimental import enable_iterative_imputer  #(necessário p/ IterativeImputer)
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from pmlb import fetch_data


# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_list(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return [s.strip() for s in value.split(",") if s.strip()]


N_JOBS = _env_int("N_JOBS", -1)
NUM_TRIALS = _env_int("NUM_TRIALS", 6)
SVR_IMPL = os.environ.get("SVR_IMPL", "linear").lower()
SVR_SCALE_Y = os.environ.get("SVR_SCALE_Y", "1") == "1"
SVR_MAX_ROWS = _env_int("SVR_MAX_ROWS", 5000)
SVR_CACHE_MB = _env_int("SVR_CACHE_MB", 1000)
RESUME = os.environ.get("RESUME", "0") == "1"

target_algorithms = _env_list("ALGS", [
    "LR",
    "KNN",
    "DT",
    "RF",
    "SVR",
    # "NN",
])

#   228_elevators e 573_house_16L existiam em versões antigas do PMLB e foram
#   removidos/renomeados no PMLB 1.0 -- fetch_data() falharia para eles hoje.
#
# Datasets sintéticos/artificiais mantidos na lista (dados de origem real
# transformados ou gerados por simulação/modelo estatístico, não medições
# diretas) -- ver documentacao_projeto.txt para a revisão completa:
#   529_pollen           -> sintético (gerado por simulação, ver metadata do PMLB)
#   225_puma8NH          -> sintético (simulação de um braço robótico Puma 560)
#   1199_BNG_echoMonths  -> sintético (réplica artificial de "echoMonths" gerada
#                           por Bayesian Network Generator, não é dado medido)
# 574_house_16H foi removido: mesma base de dados (censo de 1990) e mesmas
# 22784 linhas de 218_house_8L, só com um subconjunto de atributos escolhido
# artificialmente para ser "mais difícil" -- redundante e não agrega um
# dataset realmente novo.
#
# Lista ordenada por n_instances (menor -> maior).
datasets = _env_list("DATASETS", [
    "1089_USCrime",          # 47
    "1096_FacultySalaries",  # 50
    "542_pollution",         # 60
    "auto_insurance_losses", # 164
    "505_tecator",           # 240 (124 features -- maior razão features/instâncias da lista)
    "560_bodyfat",           # 252
    "1027_ESL",              # 488
    "1030_ERA",              # 1000
    "1028_SWD",              # 1000
    "1029_LEV",              # 1000
    "529_pollen",            # 3848  (sintético)
    "294_satellite_image",   # 6435
    "503_wind",               # 6574
    "227_cpu_small",         # 8192
    "225_puma8NH",           # 8192  (sintético)
    "197_cpu_act",           # 8192
    "201_pol",               # 15000
    "1199_BNG_echoMonths",   # 17496 (sintético)
    "537_houses",            # 20640
    "218_house_8L",          # 22784
])


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------
def _uses_kernel_svr(alg, for_tuning=False):
    """True quando o SVR é o do libsvm (custo ~O(n^2)~O(n^3))."""
    return alg == "SVR" and (for_tuning or SVR_IMPL == "libsvm")


def get_estimator(alg, for_tuning=False):
    if alg == "LR":
        return LinearRegression()
    elif alg == "KNN":
        return KNeighborsRegressor()
    elif alg == "DT":
        return DecisionTreeRegressor(random_state=42)
    elif alg == "RF":
        return RandomForestRegressor(random_state=42)
    elif alg == "SVR":
        if _uses_kernel_svr(alg, for_tuning):
            return SVR(kernel="linear", C=1, cache_size=SVR_CACHE_MB)
        # Solver liblinear: custo ~linear em n. Não é idêntico ao SVR(kernel="linear"),
        # mas resolve o mesmo problema (epsilon-insensitive, C=1, epsilon=0.1).
        return LinearSVR(
            C=1,
            epsilon=0.1,
            loss="epsilon_insensitive",
            dual=True,
            max_iter=20000,
            random_state=42,
        )
    elif alg == "NN":
        return MLPRegressor(
            hidden_layer_sizes=(100,),
            random_state=42,
            max_iter=500
        )
    raise ValueError(f"Unknown algorithm: {alg}")


def build_model(alg, *, imputer=None, selector=None, scale_x=True, for_tuning=False):
    """Monta o pipeline: [imputer] -> [selector] -> [scaler] -> regressor.

    Para o SVR o alvo também é padronizado (TransformedTargetRegressor): SVR é muito
    sensível à escala de y (C e epsilon são definidos nessa escala) e alvos grandes,
    como o preço das casas, fazem o solver demorar muito para convergir. As métricas
    continuam sendo calculadas na escala original de y.
    """
    steps = []
    if imputer is not None:
        steps.append(("imputer", imputer))
    if selector is not None:
        steps.append(("selector", selector))
    if scale_x:
        steps.append(("scaler", StandardScaler()))
    steps.append(("regressor", get_estimator(alg, for_tuning=for_tuning)))
    model = Pipeline(steps)
    if alg == "SVR" and SVR_SCALE_Y:
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    return model


def param_prefix(alg):
    """Prefixo dos hiperparâmetros do regressor dentro de build_model()."""
    return "regressor__regressor__" if (alg == "SVR" and SVR_SCALE_Y) else "regressor__"


def limit_rows(df, alg, for_tuning=False):
    """Subamostra (determinística) o dataset para o SVR com kernel, que não escala em n."""
    if _uses_kernel_svr(alg, for_tuning) and SVR_MAX_ROWS > 0 and len(df) > SVR_MAX_ROWS:
        print(f"  [SVR] subamostrando {len(df)} -> {SVR_MAX_ROWS} linhas (SVR_MAX_ROWS)", flush=True)
        return df.sample(n=SVR_MAX_ROWS, random_state=42).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def get_scoring():
    return [
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "r2"
    ]


def multiple_score(y_true, y_pred):
    return {
        "test_mae": mean_absolute_error(y_true, y_pred),
        "test_mse": mean_squared_error(y_true, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "test_r2": r2_score(y_true, y_pred),
    }


def make_dict(results):
    return {
        "test_mae":
            -results["mean_test_neg_mean_absolute_error"][
                np.argmin(results["rank_test_neg_mean_absolute_error"])],

        "test_mse":
            -results["mean_test_neg_mean_squared_error"][
                np.argmin(results["rank_test_neg_mean_squared_error"])],

        "test_rmse":
            -results["mean_test_neg_root_mean_squared_error"][
                np.argmin(results["rank_test_neg_root_mean_squared_error"])],

        "test_r2":
            results["mean_test_r2"][
                np.argmin(results["rank_test_r2"])]
    }


def make_dict_mean(results):
    return {
        "test_mae":
            -results["test_neg_mean_absolute_error"].mean(),

        "test_mse":
            -results["test_neg_mean_squared_error"].mean(),

        "test_rmse":
            -results["test_neg_root_mean_squared_error"].mean(),

        "test_r2":
            results["test_r2"].mean()
    }


# ----------------------------------------------------------------------------
# Dados faltantes
# ----------------------------------------------------------------------------
def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    # imports tardios: só o teste de imputação precisa de torch / utils
    import torch
    from utils import (
        MAR_mask,
        MNAR_mask_logistic,
        MNAR_mask_quantiles,
        MNAR_self_mask_logistic,
    )

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape, device=X.device) < p_miss).double()

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}


def get_imputer(imp):
    if imp == 'mean':
        return SimpleImputer(strategy='mean')
    elif imp == 'median':
        return SimpleImputer(strategy='median')
    elif imp == 'KNN':
        return KNNImputer(n_neighbors=5)
    elif imp == 'iterative':
        return IterativeImputer(random_state=42)
    else:
        return SimpleImputer(strategy='mean')


def insert_missing_in_df(df, perc, seed=42):
    """Insere NaNs (MAR) nas features. `seed` torna os NaNs reprodutíveis entre execuções."""
    import torch
    torch.manual_seed(seed)

    features = df.drop('target', axis=1)
    tgt = df['target'].to_numpy()
    X_miss = produce_NA(features.to_numpy(), perc, "MAR", p_obs=0.9)
    df2 = pd.DataFrame(X_miss['X_incomp'].cpu().detach().numpy(), columns=features.columns)
    df2['target'] = tgt
    return df2


# ----------------------------------------------------------------------------
# Resultados / progresso
# ----------------------------------------------------------------------------
RESULTS_DIR = "results"


def output_path(filename):
    """Monta o caminho de um CSV de resultado dentro da pasta RESULTS_DIR,
    criando a pasta se ainda não existir. Usado por cada teste para definir
    seu OUTPUT (ex.: output_path('normalization_results.csv'))."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


def load_results(path):
    """Carrega o CSV existente só quando RESUME=1; caso contrário começa vazio."""
    if RESUME and os.path.exists(path):
        print(f"[RESUME] carregando {path}", flush=True)
        return pd.read_csv(path)
    return pd.DataFrame()


def already_done(results, **keys):
    """True se já existe algum registro com todos os valores de `keys` (só com RESUME=1)."""
    if not RESUME or results.empty:
        return False
    mask = np.ones(len(results), dtype=bool)
    for col, value in keys.items():
        if col not in results.columns:
            return False
        mask &= (results[col] == value).to_numpy()
    if mask.any():
        print(f"  [RESUME] pulando {keys}", flush=True)
        return True
    return False


def append_and_save(results, rows, path):
    """Acrescenta as linhas ao DataFrame acumulado e grava o CSV (uma vez por bloco)."""
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    results.to_csv(path, index=False)
    return results


@contextmanager
def timer(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"  [{label}] {time.perf_counter() - t0:.1f}s", flush=True)
