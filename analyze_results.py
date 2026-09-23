#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_results.py
====================================================================
Le os .csv gerados pelos quatro testes de vazamento de dados
(normalization.py, imputation.py, feature_selection.py, tuning.py)
e gera uma serie de graficos comparando o cenario "com vazamento"
(leak=True) contra "sem vazamento" (leak=False), subdivididos por:

  - dataset (coluna "ds")
  - algoritmo (coluna "alg")
  - e, quando existir, a variavel especifica daquele teste:
        imputation.py         -> "imputer" e "perc-miss"
        feature_selection.py  -> "perc"
        normalization.py      -> (sem variavel extra)
        tuning.py              -> (sem variavel extra)

Para cada teste, o script produz:
  1. Boxplots comparando leak x noleak para cada metrica de regressao
     (MAE, MSE, RMSE, R2), facetados por dataset e (se existir) pela
     variavel extra do teste.
  2. Um "grafico de gap": a diferenca media (leak - noleak) de cada
     metrica, por algoritmo -- e a forma mais direta de visualizar o
     quanto o vazamento infla artificialmente a performance relatada.
  3. Uma tabela resumo (.csv) com media e desvio padrao de cada
     metrica, por grupo (ds, alg, leak, [variavel extra]).

USO
----
    python analyze_results.py
    python analyze_results.py --input-dir results --output-dir plots
    python analyze_results.py --tests normalization imputation

Por padrao, procura os quatro .csv em ./results/ (mesma pasta onde
main.py os salva) e escreve os graficos em ./plots/<teste>/ e as
tabelas resumo em ./plots/summaries/.
====================================================================
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # gera os arquivos sem precisar de tela/display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

# Nome-amigavel de cada metrica, usado nos titulos dos graficos.
METRICS = {
    "test_mae": "MAE (erro absoluto medio)",
    "test_mse": "MSE (erro quadratico medio)",
    "test_rmse": "RMSE (raiz do erro quadratico medio)",
    "test_r2": "R² (coeficiente de determinacao)",
}

# Metricas onde valores MENORES sao melhores (todas menos R²).
LOWER_IS_BETTER = {"test_mae", "test_mse", "test_rmse"}

# n_instances de cada dataset (PMLB), usado no grafico n_instancias x erro.
# Mantido fixo aqui (em vez de consultar o pmlb em tempo de analise) para o
# script rodar offline e nao depender da ordem/composicao atual da lista em
# model_utils.py. Atualize esse dicionario sempre que adicionar/remover um
# dataset em model_utils.datasets.
DATASET_SIZES = {
    "1089_USCrime": 47,
    "1096_FacultySalaries": 50,
    "542_pollution": 60,
    "auto_insurance_losses": 164,
    "505_tecator": 240,
    "560_bodyfat": 252,
    "1027_ESL": 488,
    "1030_ERA": 1000,
    "1028_SWD": 1000,
    "1029_LEV": 1000,
    "529_pollen": 3848,
    "294_satellite_image": 6435,
    "503_wind": 6574,
    "227_cpu_small": 8192,
    "225_puma8NH": 8192,
    "197_cpu_act": 8192,
    "201_pol": 15000,
    "1199_BNG_echoMonths": 17496,
    "537_houses": 20640,
    "218_house_8L": 22784,
}

# Mapeia cada teste para: nome do .csv esperado e a(s) coluna(s)
# extra(s) que subdividem os resultados alem de dataset/algoritmo.
RESULT_FILES = {
    "normalization": {
        "csv": "normalization_results.csv",
        "extra_facets": [],
    },
    "imputation": {
        "csv": "imputation_results.csv",
        "extra_facets": ["imputer", "perc-miss"],
    },
    "feature_selection": {
        "csv": "feature_selection_results.csv",
        "extra_facets": ["perc"],
    },
    "tuning": {
        "csv": "hyperparameter_tuning_results.csv",
        "extra_facets": [],
    },
}


def load_csv(path):
    """Carrega um .csv de resultados. Devolve None (com aviso) se o
    arquivo nao existir ou vier vazio -- assim o script nao quebra
    caso algum teste ainda nao tenha sido rodado."""
    if not os.path.exists(path):
        print(f"  [aviso] arquivo nao encontrado, pulando: {path}")
        return None

    df = pd.read_csv(path)
    if df.empty:
        print(f"  [aviso] arquivo vazio, pulando: {path}")
        return None

    # Normaliza para rotulos legiveis nos graficos.
    if "leak" in df.columns:
        df["leak"] = df["leak"].astype(str).map(
            {"True": "Com vazamento", "False": "Sem vazamento"}
        ).fillna(df["leak"])

    return df


def _facet_kwargs(df, extra_facet):
    """Monta os kwargs de linha/coluna do seaborn.catplot: usa a
    variavel extra do teste (imputer, perc, perc-miss) como coluna
    e o dataset como linha, apenas quando cada uma tiver mais de um
    valor distinto (evita facetas com um unico grafico redundante).
    col_wrap so pode ser usado quando NAO ha faceta "row" (limitacao
    do seaborn), por isso so entra no caso de uma unica dimensao
    extra."""
    kwargs = {}
    has_extra = extra_facet and extra_facet in df.columns and df[extra_facet].nunique() > 1
    has_ds = "ds" in df.columns and df["ds"].nunique() > 1

    if has_extra and has_ds:
        kwargs["col"] = extra_facet
        kwargs["row"] = "ds"
    elif has_extra:
        kwargs["col"] = extra_facet
        kwargs["col_wrap"] = min(4, df[extra_facet].nunique())
    elif has_ds:
        kwargs["col"] = "ds"
        kwargs["col_wrap"] = min(4, df["ds"].nunique())
    return kwargs


def plot_metric_boxplots(df, metric, test_name, outdir, extra_facet=None):
    """Boxplot de `metric` por algoritmo, comparando leak x noleak,
    facetado por dataset e (se aplicavel) pela variavel extra do
    teste. Um arquivo .png por metrica (por variavel extra)."""
    if metric not in df.columns:
        return

    kwargs = _facet_kwargs(df, extra_facet)
    g = sns.catplot(
        data=df, kind="box",
        x="alg", y=metric, hue="leak",
        order=sorted(df["alg"].unique()),
        height=4, aspect=1.15,
        **kwargs,
    )
    g.set_axis_labels("Algoritmo", METRICS.get(metric, metric))
    g.legend.set_title("Cenario")
    subtitle = f" (por {extra_facet})" if extra_facet else ""
    g.fig.suptitle(f"{test_name} -- {METRICS.get(metric, metric)}{subtitle}", y=1.03)
    g.tight_layout()

    suffix = f"_por_{extra_facet.replace('-', '')}" if extra_facet else ""
    fname = f"{test_name}_{metric}{suffix}.png"
    path = os.path.join(outdir, fname)
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  salvo: {path}")

#Deixei por precaução, mas é uma representação redundante do plot_metric_boxplots, que é mais completa e flexível em geral.
def plot_leakage_gap(df, metric, test_name, outdir, extra_facet=None):
    """Grafico de barras com a diferenca MEDIA (leak - noleak) da
    metrica, por algoritmo. E o grafico mais direto para responder
    'o quanto o vazamento infla a performance reportada' -- valores
    diferentes de zero mostram o tamanho do vies introduzido.
    Para MAE/MSE/RMSE (menor é melhor), gap negativo = vazamento fez
    o erro parecer MENOR do que realmente é (performance inflada).
    Para R² (maior é melhor), gap positivo = vazamento fez o R²
    parecer MAIOR do que realmente é (performance inflada).
    """
    if metric not in df.columns:
        return

    group_cols = ["alg"]
    if extra_facet and extra_facet in df.columns and df[extra_facet].nunique() > 1:
        group_cols.append(extra_facet)
    if "ds" in df.columns and df["ds"].nunique() > 1:
        group_cols.append("ds")

    means = (
        df.groupby(group_cols + ["leak"])[metric]
        .mean()
        .unstack("leak")
    )
    if "Com vazamento" not in means.columns or "Sem vazamento" not in means.columns:
        return  # faltou um dos dois cenarios nesse recorte, nao da pra comparar

    means["gap"] = means["Com vazamento"] - means["Sem vazamento"]
    means = means.reset_index()

    plt.figure(figsize=(max(6, 1.4 * means["alg"].nunique() * max(1, len(group_cols))), 5))
    hue = group_cols[1] if len(group_cols) > 1 else None
    ax = sns.barplot(data=means, x="alg", y="gap", hue=hue)
    ax.axhline(0, color="black", linewidth=1)

    direction = "menor é melhor" if metric in LOWER_IS_BETTER else "maior é melhor"
    ax.set_xlabel("Algoritmo")
    ax.set_ylabel(f"gap = média(com vazamento) - média(sem vazamento)")
    ax.set_title(
        f"{test_name} -- vies de vazamento em {METRICS.get(metric, metric)} ({direction})"
    )
    plt.tight_layout()

    suffix = f"_por_{extra_facet.replace('-', '')}" if hue == extra_facet else ""
    fname = f"{test_name}_{metric}_gap{suffix}.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {path}")


def plot_error_vs_size(df, metric, test_name, outdir):
    """Grafico de dispersao/linha: n_instances do dataset (escala log, eixo x)
    contra a metrica de erro (eixo y), comparando leak x noleak. Cada ponto
    e a media da metrica para um dataset+cenario (media entre algoritmos e
    repeticoes). O objetivo e visualizar se o vies do vazamento (a distancia
    entre as duas linhas) fica maior em datasets com poucas instancias.
    Datasets sem tamanho conhecido em DATASET_SIZES sao ignorados."""
    if metric not in df.columns or "ds" not in df.columns:
        return

    sizes = df["ds"].map(DATASET_SIZES)
    if sizes.isna().all():
        print(f"  [aviso] nenhum dataset de {test_name} tem tamanho conhecido em DATASET_SIZES, pulando grafico de tamanho")
        return

    plot_df = df.assign(n_instances=sizes).dropna(subset=["n_instances"])
    means = (
        plot_df.groupby(["ds", "n_instances", "leak"])[metric]
        .mean()
        .reset_index()
        .sort_values("n_instances")
    )

    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=means, x="n_instances", y=metric, hue="leak",
        marker="o", sort=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Numero de instancias do dataset (escala log)")
    ax.set_ylabel(METRICS.get(metric, metric))
    ax.set_title(f"{test_name} -- {METRICS.get(metric, metric)} x tamanho do dataset")
    if ax.legend_ is not None:
        ax.legend_.set_title("Cenario")
    plt.tight_layout()

    fname = f"{test_name}_{metric}_por_tamanho.png"
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {path}")


def save_summary_table(df, test_name, extra_facets, summaries_dir):
    """Salva uma tabela .csv com media e desvio padrao de cada
    metrica, agrupada por dataset/algoritmo/leak/(variaveis extras).
    Util para conferir os numeros exatos por tras dos graficos."""
    group_cols = [c for c in (["ds", "alg", "leak"] + extra_facets) if c in df.columns]
    metric_cols = [m for m in METRICS if m in df.columns]
    if not metric_cols:
        return

    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    path = os.path.join(summaries_dir, f"{test_name}_summary.csv")
    summary.to_csv(path, index=False)
    print(f"  salvo: {path}")


def analyze_test(test_name, cfg, input_dir, output_root, summaries_dir):
    print(f"\n=== {test_name} ===")
    csv_path = os.path.join(input_dir, cfg["csv"])
    df = load_csv(csv_path)
    if df is None:
        return

    outdir = os.path.join(output_root, test_name)
    os.makedirs(outdir, exist_ok=True)

    save_summary_table(df, test_name, cfg["extra_facets"], summaries_dir)

    for metric in METRICS:
        plot_metric_boxplots(df, metric, test_name, outdir)
        plot_leakage_gap(df, metric, test_name, outdir)
        plot_error_vs_size(df, metric, test_name, outdir)
        for facet in cfg["extra_facets"]:
            plot_metric_boxplots(df, metric, test_name, outdir, extra_facet=facet)
            plot_leakage_gap(df, metric, test_name, outdir, extra_facet=facet)


def main():
    parser = argparse.ArgumentParser(
        description="Gera graficos e tabelas resumo a partir dos .csv de resultado "
                    "dos testes de vazamento de dados."
    )
    parser.add_argument(
        "--input-dir", default="results",
        help="Diretorio onde estao os .csv de resultado (padrao: ./results, "
             "pasta onde main.py agora grava os CSVs)."
    )
    parser.add_argument(
        "--output-dir", default="plots",
        help="Diretorio onde os graficos/tabelas serao salvos (padrao: ./plots)."
    )
    parser.add_argument(
        "--tests", nargs="+", choices=list(RESULT_FILES.keys()),
        default=list(RESULT_FILES.keys()),
        help="Quais testes processar (padrao: todos)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summaries_dir = os.path.join(args.output_dir, "summaries")
    os.makedirs(summaries_dir, exist_ok=True)

    for test_name in args.tests:
        analyze_test(test_name, RESULT_FILES[test_name], args.input_dir, args.output_dir, summaries_dir)

    print(f"\nConcluido. Graficos em '{args.output_dir}/<teste>/' e tabelas resumo em '{summaries_dir}/'.")


if __name__ == "__main__":
    main()
