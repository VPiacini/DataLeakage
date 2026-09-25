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
  3. Um grafico de linha da propria metrica (nao o delta) x tamanho do
     dataset, comparando leak x noleak diretamente (plot_error_vs_size).
  4. Dois graficos de "efeito x tamanho": dispersao do delta (com - sem
     vazamento) de cada metrica contra o numero de instancias do dataset
     (escala log), com uma tendencia media (parabola, grau <= 2) ajustada
     em escala symlog -- uma versao com um ponto por par dataset+algoritmo
     (plot_leakage_delta_vs_size_por_alg) e outra com um ponto por
     dataset, ja agregando os algoritmos (plot_leakage_delta_vs_size_por_dataset).
     Datasets ausentes de DATASET_SIZES sao ignorados nesses dois graficos.
  5. Uma tabela resumo (.csv) com media e desvio padrao de cada
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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.scale import SymmetricalLogTransform
from scipy.stats import spearmanr

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


def _paired_delta_by_size(df, metric, extra_facets):
    """Pareia leak/noleak por (ds, alg, iteration, extra_facets) e calcula
    o delta (com vazamento - sem vazamento) de `metric` para cada
    combinacao -- o mesmo tipo de pareamento usado nos graficos de "gap",
    mas mantendo uma linha por combinacao (em vez de ja agregar por
    algoritmo). Mapeia o tamanho de cada dataset via DATASET_SIZES e
    descarta datasets sem tamanho conhecido. Devolve None se faltar
    algum dos dois cenarios ou nenhum dataset tiver tamanho conhecido."""
    if metric not in df.columns or "ds" not in df.columns or "alg" not in df.columns:
        return None
    if "leak" not in df.columns or "iteration" not in df.columns:
        return None

    id_cols = [c for c in (["ds", "alg", "iteration"] + extra_facets) if c in df.columns]
    piv = df.pivot_table(index=id_cols, columns="leak", values=metric, aggfunc="first")
    if "Com vazamento" not in piv.columns or "Sem vazamento" not in piv.columns:
        return None

    piv = piv.reset_index()
    piv["delta"] = piv["Com vazamento"] - piv["Sem vazamento"]
    piv["n_instances"] = piv["ds"].map(DATASET_SIZES)
    piv = piv.dropna(subset=["n_instances", "delta"])
    return piv if not piv.empty else None


def _symlog_quad_trend(x, y):
    """Ajusta uma tendencia MEDIA (parabola, grau <= 2, minimos quadrados)
    entre log10(x) e y, trabalhando no mesmo espaco "symlog" em que o eixo
    y sera desenhado -- assim a curva aparece como uma parabola de fato no
    grafico, mesmo com y passando perto de zero. O limiar da regiao linear
    do symlog (linthresh) e o percentil 10 dos valores absolutos NAO-NULOS
    de y: ancora a regiao linear nos menores efeitos observados e deixa a
    maior parte dos pontos na regiao log. Devolve (xs, ys, linthresh),
    prontos para plt.plot() e para configurar o proprio eixo."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    nz = np.abs(y[y != 0])
    linthresh = max(np.percentile(nz, 10), 1e-12) if nz.size else 1e-6

    tr = SymmetricalLogTransform(10, linthresh, 1)
    deg = min(2, len(np.unique(x)) - 1)
    coef = np.polyfit(np.log10(x), tr.transform(y), deg)
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ys = tr.inverted().transform(np.polyval(coef, np.log10(xs)))
    return xs, ys, linthresh


def plot_leakage_delta_vs_size_por_alg(df, metric, test_name, outdir, extra_facets):
    """Grafico "efeito x tamanho" (versao 1): dispersao do delta (com -
    sem vazamento) de `metric`, um ponto por par (dataset, algoritmo) --
    media entre iteracoes e niveis extras (perc, imputer, perc-miss).
    Colorido por algoritmo. Eixo x = n_instancias (escala log), eixo y em
    escala symlog (os efeitos variam de ~1e-6 a ~1e-1). A linha preta e
    uma tendencia media (parabola, grau <= 2) ajustada no espaco symlog do
    eixo y. O texto no canto mostra a correlacao de Spearman entre o
    delta e n_instancias, como referencia rapida da forca da relacao."""
    piv = _paired_delta_by_size(df, metric, extra_facets)
    if piv is None:
        return

    g = (piv.groupby(["ds", "alg"], as_index=False)
            .agg(delta=("delta", "mean"), n_instances=("n_instances", "first")))
    if g["ds"].nunique() < 4:
        print(f"  [aviso] poucos datasets com tamanho conhecido para {test_name}/{metric}, pulando grafico de tamanho")
        return

    algs = sorted(g["alg"].unique())
    palette = dict(zip(algs, sns.color_palette("deep", len(algs))))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for a in algs:
        s = g[g["alg"] == a]
        ax.scatter(s["n_instances"], s["delta"], label=a, color=palette[a],
                   s=45, alpha=.85, edgecolor="white", linewidth=.5, zorder=3)

    xs, ys, linthresh = _symlog_quad_trend(g["n_instances"], g["delta"])
    ax.plot(xs, ys, color="black", lw=2.2, zorder=4, label="Tendencia (ajuste quadratico)")
    ax.axhline(0, color="grey", lw=.8, zorder=1)

    rho, p = spearmanr(g["delta"], g["n_instances"])
    ax.text(.97, .97, f"Spearman rho = {rho:+.2f}\np = {p:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="lightgrey"))

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel("Numero de instancias do dataset (escala log)")
    ax.set_ylabel(f"delta {METRICS.get(metric, metric)}\n(com - sem vazamento) [escala symlog]")
    ax.set_title(f"{test_name} -- efeito do vazamento x tamanho do dataset (por algoritmo)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.32), ncol=min(6, len(algs) + 1), frameon=False)
    fig.tight_layout()

    fname = f"{test_name}_{metric}_delta_vs_tamanho_alg.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {path}")


def plot_leakage_delta_vs_size_por_dataset(df, metric, test_name, outdir, extra_facets):
    """Grafico "efeito x tamanho" (versao 2): mesma ideia da funcao
    anterior, mas com um unico ponto por dataset -- media do delta entre
    TODOS os algoritmos, iteracoes e niveis extras daquele dataset. Usa
    MEDIA (nao mediana): em testes onde a maioria dos pares empata em
    delta=0 (ex.: selecao de atributos, onde ~90% dos pares empatam), a
    mediana por dataset ficaria sempre zero e esconderia o efeito, que so
    aparece em alguns poucos datasets/algoritmos."""
    piv = _paired_delta_by_size(df, metric, extra_facets)
    if piv is None:
        return

    g = (piv.groupby("ds", as_index=False)
            .agg(delta=("delta", "mean"), n_instances=("n_instances", "first")))
    if len(g) < 4:
        print(f"  [aviso] poucos datasets com tamanho conhecido para {test_name}/{metric}, pulando grafico de tamanho")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(g["n_instances"], g["delta"], s=55, color="#4C72B0", alpha=.85,
               edgecolor="white", linewidth=.5, zorder=3)

    xs, ys, linthresh = _symlog_quad_trend(g["n_instances"], g["delta"])
    ax.plot(xs, ys, color="black", lw=2.2, zorder=4, label="Tendencia (ajuste quadratico)")
    ax.axhline(0, color="grey", lw=.8, zorder=1)

    rho, p = spearmanr(g["delta"], g["n_instances"])
    ax.text(.97, .97, f"Spearman rho = {rho:+.2f}\np = {p:.3g}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="lightgrey"))

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel("Numero de instancias do dataset (escala log)")
    ax.set_ylabel(f"delta {METRICS.get(metric, metric)} por dataset\n(media entre algoritmos) [escala symlog]")
    ax.set_title(f"{test_name} -- efeito do vazamento x tamanho do dataset (por dataset)")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, -.22), frameon=False)
    fig.tight_layout()

    fname = f"{test_name}_{metric}_delta_vs_tamanho_dataset.png"
    path = os.path.join(outdir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
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
        plot_leakage_delta_vs_size_por_alg(df, metric, test_name, outdir, cfg["extra_facets"])
        plot_leakage_delta_vs_size_por_dataset(df, metric, test_name, outdir, cfg["extra_facets"])
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
