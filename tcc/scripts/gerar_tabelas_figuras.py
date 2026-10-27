#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera as tabelas (.tex) e figuras (.pdf) usadas no texto do TCC a partir dos
CSVs brutos em results/. Todos os numeros citados nos capitulos 4 e 5 saem
deste script, de modo que o texto pode ser regenerado se os experimentos
forem refeitos.

Definicoes (as mesmas dos CSVs de analise *_pares.csv):
  delta_r2   = R2(com vazamento) - R2(sem vazamento)
  delta_rmse = 100 * (RMSE(sem) - RMSE(com)) / RMSE(sem)   [% de reducao]
Em ambos, valor positivo = vazamento fez o modelo parecer melhor.

Uso (a partir da raiz do repositorio):
    python tcc/scripts/gerar_tabelas_figuras.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
TCC = os.path.join(ROOT, "tcc")
TAB = os.path.join(TCC, "tabelas")
FIG = os.path.join(TCC, "figuras")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TIE_TOL = 1e-9   # |delta_r2| abaixo disso conta como empate
ALGS = ["LR", "KNN", "DT", "RF", "SVR"]

SCENARIOS = {
    "normalization": ("normalization_results.csv", []),
    "imputation": ("imputation_results.csv", ["imputer", "perc-miss"]),
    "feature_selection": ("feature_selection_results.csv", ["perc"]),
    "tuning": ("hyperparameter_tuning_results.csv", []),
}
SCEN_LABEL = {
    "normalization": "Normalization",
    "imputation": "Imputation",
    "feature_selection": "Feature selection",
    "tuning": "Hyperparameter tuning",
}

# Caracteristicas dos datasets (PMLB). Dominio conforme metadados do PMLB /
# descricao original de cada base.
DATASETS = {
    "1089_USCrime": (47, 13, "Criminology"),
    "1096_FacultySalaries": (50, 4, "Economics"),
    "542_pollution": (60, 15, "Environmental health"),
    "505_tecator": (240, 124, "Food spectroscopy"),
    "560_bodyfat": (252, 14, "Human physiology"),
    "1027_ESL": (488, 4, "Human resources"),
    "1028_SWD": (1000, 10, "Social services"),
    "1029_LEV": (1000, 4, "Education"),
    "1030_ERA": (1000, 4, "Human resources"),
    "529_pollen": (3848, 4, "Botany (synthetic)"),
    "294_satellite_image": (6435, 36, "Remote sensing"),
    "503_wind": (6574, 14, "Meteorology"),
    "197_cpu_act": (8192, 21, "Computer systems"),
    "225_puma8NH": (8192, 8, "Robotics (synthetic)"),
    "227_cpu_small": (8192, 12, "Computer systems"),
    "201_pol": (15000, 48, "Telecommunications"),
    "1199_BNG_echoMonths": (17496, 9, "Healthcare (synthetic)"),
    "537_houses": (20640, 8, "Real estate"),
    "218_house_8L": (22784, 8, "Real estate"),
    "574_house_16H": (22784, 16, "Real estate"),
    "564_fried": (40768, 10, "Synthetic benchmark"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
ALG_COLORS = dict(zip(ALGS, ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]))


# ---------------------------------------------------------------------------
# Utilitarios
# ---------------------------------------------------------------------------
def ds_tex(name):
    return name.replace("_", r"\_")


def fmt(x, nd=4):
    if pd.isna(x):
        return "--"
    if x != 0 and abs(x) < 10 ** (-nd):
        mant, exp = f"{x:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:.{nd}f}"


def fmt_p(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    if np.all(np.abs(x) < TIE_TOL):
        return 1.0
    try:
        return wilcoxon(x).pvalue
    except ValueError:
        return 1.0


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def load_pairs(name):
    csv, extra = SCENARIOS[name]
    df = pd.read_csv(os.path.join(RESULTS, csv))
    df["leak"] = df["leak"].astype(str)
    keys = ["ds", "alg", "iteration"] + extra
    piv = df.pivot_table(index=keys, columns="leak",
                         values=["test_r2", "test_rmse", "test_mae"], aggfunc="first")
    piv.columns = [f"{m}_{'leak' if l == 'True' else 'noleak'}" for m, l in piv.columns]
    piv = piv.reset_index()
    piv["delta_r2"] = piv["test_r2_leak"] - piv["test_r2_noleak"]
    piv["delta_rmse"] = 100 * (piv["test_rmse_noleak"] - piv["test_rmse_leak"]) / piv["test_rmse_noleak"]
    piv["delta_mae"] = 100 * (piv["test_mae_noleak"] - piv["test_mae_leak"]) / piv["test_mae_noleak"]
    piv["n_instances"] = piv["ds"].map(lambda d: DATASETS[d][0])
    piv["n_features"] = piv["ds"].map(lambda d: DATASETS[d][1])
    return piv


def alg_summary(pairs, group_extra=()):
    """Resumo por algoritmo (e niveis extras): estatisticas por dataset
    (media dos pares de cada dataset) + percentual de pares melhor/empate/pior,
    Wilcoxon sobre as 21 medias por dataset e correcao de Holm entre algoritmos
    (dentro de cada nivel extra)."""
    rows = []
    gcols = list(group_extra) + ["alg"]
    for key, s in pairs.groupby(gcols):
        key = key if isinstance(key, tuple) else (key,)
        per_ds = s.groupby("ds")[["delta_r2", "delta_rmse"]].mean()
        d = s["delta_r2"]
        rows.append({
            **dict(zip(gcols, key)),
            "n_ds": len(per_ds),
            "mean_r2": per_ds["delta_r2"].mean(),
            "median_r2": per_ds["delta_r2"].median(),
            "max_r2": per_ds["delta_r2"].max(),
            "mean_rmse": per_ds["delta_rmse"].mean(),
            "median_rmse": per_ds["delta_rmse"].median(),
            "max_rmse": per_ds["delta_rmse"].max(),
            "pct_better": 100 * (d > TIE_TOL).mean(),
            "pct_tie": 100 * (d.abs() <= TIE_TOL).mean(),
            "pct_worse": 100 * (d < -TIE_TOL).mean(),
            "p_r2": safe_wilcoxon(per_ds["delta_r2"]),
            "p_rmse": safe_wilcoxon(per_ds["delta_rmse"]),
        })
    out = pd.DataFrame(rows)
    out["alg"] = pd.Categorical(out["alg"], ALGS, ordered=True)
    if group_extra:
        out["p_holm_r2"] = out.groupby(list(group_extra))["p_r2"].transform(lambda p: holm(p.values))
        out["p_holm_rmse"] = out.groupby(list(group_extra))["p_rmse"].transform(lambda p: holm(p.values))
    else:
        out["p_holm_r2"] = holm(out["p_r2"].values)
        out["p_holm_rmse"] = holm(out["p_rmse"].values)
    return out.sort_values(list(group_extra) + ["alg"]).reset_index(drop=True)


def ds_alg_summary(pairs, group_extra=()):
    gcols = ["ds", "alg"] + list(group_extra)
    rows = []
    for key, s in pairs.groupby(gcols):
        d = s["delta_r2"]
        rows.append({**dict(zip(gcols, key)),
                     "mean_r2": d.mean(), "mean_rmse": s["delta_rmse"].mean(),
                     "all_better": bool((d > TIE_TOL).all()),
                     "all_tie": bool((d.abs() <= TIE_TOL).all()),
                     "p": safe_wilcoxon(d)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tabelas LaTeX
# ---------------------------------------------------------------------------
def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("  salvo:", os.path.relpath(path, ROOT))


def table_datasets():
    lines = [r"\begin{tabular}{lrrrl}", r"\toprule",
             r"\textbf{Data set} & \textbf{Instances} & \textbf{Features} & \textbf{$d/n$} & \textbf{Domain} \\",
             r"\midrule"]
    for ds, (n, f, dom) in sorted(DATASETS.items(), key=lambda kv: (kv[1][0], kv[0])):
        lines.append(f"{ds_tex(ds)} & {n:,} & {f} & {f / n:.4f} & {dom} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(TAB, "tab_datasets.tex"), "\n".join(lines) + "\n")


def table_alg(summary, fname, first_cols=()):
    head_extra = "".join(f"\\textbf{{{c}}} & " for c, _ in first_cols)
    ncols = len(first_cols)
    spec = "l" * ncols + "lrrrrrrrr"
    lines = [f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
             head_extra + r"\textbf{Alg.} & \multicolumn{3}{c}{\textbf{$\Delta R^2$ per data set}} & "
             r"\textbf{$\Delta$RMSE (\%)} & \textbf{Better} & \textbf{Tie} & \textbf{Worse} & \textbf{$p_{\text{Holm}}$} \\",
             f"\\cmidrule(lr){{{ncols + 2}-{ncols + 4}}}",
             "& " * ncols + r"& Mean & Median & Max & Mean & (\%) & (\%) & (\%) & ($R^2$) \\",
             r"\midrule"]
    prev = None
    for _, r in summary.iterrows():
        lead = ""
        for c, fn in first_cols:
            val = fn(r[c]) if fn else r[c]
            lead += f"{val} & "
        cur = tuple(r[c] for c, _ in first_cols)
        if prev is not None and cur != prev and ncols:
            lines.append(r"\midrule")
        prev = cur
        lines.append(
            lead + f"{r['alg']} & {fmt(r['mean_r2'])} & {fmt(r['median_r2'])} & {fmt(r['max_r2'])} & "
            f"{r['mean_rmse']:.2f} & {r['pct_better']:.1f} & {r['pct_tie']:.1f} & {r['pct_worse']:.1f} & "
            f"{fmt_p(r['p_holm_r2'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(TAB, fname), "\n".join(lines) + "\n")


def table_top(dsalg, scen, k=8, extra=()):
    top = dsalg.sort_values("mean_r2", ascending=False).head(k)
    head = r"\textbf{Data set} & \textbf{Alg.} & " + "".join(f"\\textbf{{{e}}} & " for e in extra) + \
        r"\textbf{$\Delta R^2$} & \textbf{$\Delta$RMSE (\%)} \\"
    lines = [r"\begin{tabular}{ll" + "r" * len(extra) + "rr}", r"\toprule", head, r"\midrule"]
    for _, r in top.iterrows():
        ex = "".join(f"{r[e]} & " for e in extra)
        lines.append(f"{ds_tex(r['ds'])} & {r['alg']} & {ex}{r['mean_r2']:.4f} & {r['mean_rmse']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(TAB, f"tab_top_{scen}.tex"), "\n".join(lines) + "\n")


def table_overview(overview):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"\textbf{Scenario} & \multicolumn{3}{c}{\textbf{$\Delta R^2$}} & \textbf{$\Delta$RMSE (\%)} & "
             r"\textbf{Better} & \textbf{Tie} \\",
             r"\cmidrule(lr){2-4}",
             r"& Mean & Median & Max & Mean & (\%) & (\%) \\",
             r"\midrule"]
    for _, r in overview.iterrows():
        lines.append(f"{SCEN_LABEL[r['scenario']]} & {fmt(r['mean_r2'])} & {fmt(r['median_r2'])} & "
                     f"{fmt(r['max_r2'])} & {r['mean_rmse']:.2f} & {r['pct_better']:.1f} & {r['pct_tie']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(os.path.join(TAB, "tab_overview.tex"), "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_heatmap(pairs, scen, fname, title=None):
    m = pairs.groupby(["ds", "alg"])["delta_r2"].mean().unstack("alg")[ALGS]
    order = sorted(m.index, key=lambda d: DATASETS[d][0])
    m = m.loc[order]
    vmax = np.nanmax(np.abs(m.values))
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    lim = max(vmax, 1e-6)
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(ALGS)), ALGS)
    ax.set_yticks(range(len(order)), [f"{d} ({DATASETS[d][0]})" for d in order])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if abs(v) >= 0.0005:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 * lim else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"Mean $\Delta R^2$ (leak $-$ no leak)")
    ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), bbox_inches="tight")
    plt.close(fig)
    print("  salvo:", os.path.relpath(os.path.join(FIG, fname), ROOT))


def fig_overview_box(all_pairs):
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.6), sharey=False)
    for ax, (scen, pairs) in zip(axes, all_pairs.items()):
        per = pairs.groupby(["ds", "alg"])["delta_r2"].mean().reset_index()
        data = [per.loc[per.alg == a, "delta_r2"].values for a in ALGS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                        flierprops=dict(markersize=2.5, alpha=.6))
        for patch, a in zip(bp["boxes"], ALGS):
            patch.set_facecolor(ALG_COLORS[a])
            patch.set_alpha(.75)
        for med in bp["medians"]:
            med.set_color("black")
        ax.axhline(0, color="grey", lw=.7, zorder=0)
        ax.set_xticks(range(1, 6), ALGS, rotation=90)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(SCEN_LABEL[scen])
    axes[0].set_ylabel(r"Mean $\Delta R^2$ per data set (symlog)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_overview_box.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  salvo: tcc/figuras/fig_overview_box.pdf")


def fig_size(all_pairs):
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.4), sharex=True)
    out = {}
    for ax, (scen, pairs) in zip(axes, all_pairs.items()):
        per = pairs.groupby(["ds", "alg"]).agg(d=("delta_r2", "mean"), n=("n_instances", "first")).reset_index()
        for a in ALGS:
            s = per[per.alg == a]
            ax.scatter(s["n"], s["d"], s=9, color=ALG_COLORS[a], label=a, alpha=.85, edgecolor="none")
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.axhline(0, color="grey", lw=.7, zorder=0)
        ax.set_title(SCEN_LABEL[scen])
        ax.set_xlabel("Instances (log)")
        per_ds = per.groupby("ds").agg(d=("d", "mean"), n=("n", "first"))
        rho, p = spearmanr(per_ds["n"], per_ds["d"])
        out[scen] = (rho, p)
        ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(.97, .97, f"$\\rho$={rho:+.2f}\n{ptxt}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)
    axes[0].set_ylabel(r"Mean $\Delta R^2$ (symlog)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, -.08))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_delta_vs_size.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  salvo: tcc/figuras/fig_delta_vs_size.pdf")
    return out


def fig_fs_perc(pairs):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    m = pairs.groupby(["perc", "alg"])["delta_r2"].mean().unstack("alg")[ALGS]
    for a in ALGS:
        ax.plot(m.index, m[a], marker="o", ms=3.5, color=ALG_COLORS[a], label=a)
    ax.axhline(0, color="grey", lw=.7)
    ax.set_xticks(m.index, [f"{p}%" for p in m.index])
    ax.set_xlabel("Percentile of features retained")
    ax.set_ylabel(r"Mean $\Delta R^2$")
    ax.legend(ncol=5, fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fs_perc.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  salvo: tcc/figuras/fig_fs_perc.pdf")


def fig_imp(pairs):
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4), sharey=True)
    for ax, imp in zip(axes, ["mean", "median", "KNN"]):
        s = pairs[pairs.imputer == imp]
        m = s.groupby(["perc-miss", "alg"])["delta_r2"].mean().unstack("alg")[ALGS]
        for a in ALGS:
            ax.plot(100 * m.index, m[a], marker="o", ms=3, color=ALG_COLORS[a], label=a)
        ax.axhline(0, color="grey", lw=.7)
        ax.set_title(f"{imp.capitalize() if imp != 'KNN' else 'KNN'} imputer")
        ax.set_xticks([5, 10, 20, 30])
        ax.set_xlabel("Missing rate (%)")
    axes[0].set_ylabel(r"Mean $\Delta R^2$")
    axes[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_imp_rate.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  salvo: tcc/figuras/fig_imp_rate.pdf")


def fig_dataset_profile():
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for ds, (n, f, dom) in DATASETS.items():
        syn = "synthetic" in dom.lower()
        ax.scatter(n, f, s=22, marker="^" if syn else "o",
                   color="#C44E52" if syn else "#4C72B0", edgecolor="none", alpha=.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of instances (log)")
    ax.set_ylabel("Number of features (log)")
    ax.scatter([], [], marker="o", color="#4C72B0", label="Real-world")
    ax.scatter([], [], marker="^", color="#C44E52", label="Synthetic")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_datasets.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  salvo: tcc/figuras/fig_datasets.pdf")


# ---------------------------------------------------------------------------
def main():
    stats = {}
    table_datasets()
    fig_dataset_profile()

    all_pairs = {s: load_pairs(s) for s in SCENARIOS}

    # Visao geral por cenario
    ov = []
    for scen, p in all_pairs.items():
        per = p.groupby(["ds", "alg"])[["delta_r2", "delta_rmse"]].mean()
        ov.append({"scenario": scen, "mean_r2": per["delta_r2"].mean(), "median_r2": per["delta_r2"].median(),
                   "max_r2": per["delta_r2"].max(), "mean_rmse": per["delta_rmse"].mean(),
                   "pct_better": 100 * (p["delta_r2"] > TIE_TOL).mean(),
                   "pct_tie": 100 * (p["delta_r2"].abs() <= TIE_TOL).mean(),
                   "n_pairs": len(p)})
    ov = pd.DataFrame(ov)
    table_overview(ov)
    stats["overview"] = ov.to_dict(orient="records")

    for scen, pairs in all_pairs.items():
        extra = SCENARIOS[scen][1]
        summ = alg_summary(pairs)
        table_alg(summ, f"tab_alg_{scen}.tex")
        dsalg = ds_alg_summary(pairs, extra)
        stats[scen] = {
            "alg": summ.assign(alg=summ.alg.astype(str)).to_dict(orient="records"),
            "n_pairs": len(pairs),
            "n_configs": len(dsalg),
            "configs_all_better": int(dsalg["all_better"].sum()),
            "configs_all_tie": int(dsalg["all_tie"].sum()),
            "configs_p05": int((dsalg["p"] < 0.05).sum()),
        }
        # Top configuracoes (media sobre niveis extras)
        dsalg_mean = ds_alg_summary(pairs)
        table_top(dsalg_mean, scen)
        stats[scen]["top"] = dsalg_mean.sort_values("mean_r2", ascending=False).head(10).to_dict(orient="records")
        # Friedman entre algoritmos (bloco = dataset)
        m = pairs.groupby(["ds", "alg"])["delta_r2"].mean().unstack("alg")[ALGS]
        fr = friedmanchisquare(*[m[a].values for a in ALGS])
        stats[scen]["friedman"] = {"stat": fr.statistic, "p": fr.pvalue,
                                   "mean_rank": m.rank(axis=1, ascending=False).mean().to_dict()}
        # Correlacoes com caracteristicas dos datasets
        per_ds = pairs.groupby("ds").agg(d=("delta_r2", "mean"), n=("n_instances", "first"),
                                         f=("n_features", "first"))
        per_ds["ratio"] = per_ds.f / per_ds.n
        stats[scen]["spearman"] = {c: dict(zip(["rho", "p"], spearmanr(per_ds[c], per_ds["d"])))
                                   for c in ["n", "f", "ratio"]}
        small = pairs[pairs.n_instances < 1000]
        large = pairs[pairs.n_instances >= 1000]
        stats[scen]["size_groups"] = {
            g: {"n_ds": int(x.ds.nunique()), "mean_r2": x.delta_r2.mean(), "mean_abs_r2": x.delta_r2.abs().mean(),
                "mean_rmse": x.delta_rmse.mean()}
            for g, x in [("lt1000", small), ("ge1000", large)]}
        stats[scen]["per_ds"] = per_ds.sort_values("d", ascending=False)["d"].to_dict()
        fig_heatmap(pairs, scen, f"fig_heatmap_{scen}.pdf")

        # Detalhamento por nivel
        if scen == "feature_selection":
            s2 = alg_summary(pairs, ["perc"])
            table_alg(s2, "tab_alg_fs_perc.tex", first_cols=[("perc", lambda v: f"{int(v)}\\%")])
            stats[scen]["by_perc"] = s2.assign(alg=s2.alg.astype(str)).to_dict(orient="records")
            stats[scen]["by_perc_overall"] = pairs.groupby("perc")["delta_r2"].agg(
                ["mean", lambda x: 100 * (x.abs() <= TIE_TOL).mean()]).to_dict()
            # numero de atributos efetivamente selecionados
            nsel = {ds: {p: max(1, int(round(f * max(p, 100 / f) / 100))) for p in [1, 5, 10, 20]}
                    for ds, (n, f, _) in DATASETS.items()}
            stats[scen]["n_selected_approx"] = nsel
            fig_fs_perc(pairs)
        if scen == "imputation":
            s_imp = alg_summary(pairs, ["imputer"])
            s_rate = alg_summary(pairs, ["perc-miss"])
            table_alg(s_imp, "tab_alg_imp_imputer.tex", first_cols=[("imputer", None)])
            stats[scen]["by_imputer"] = s_imp.assign(alg=s_imp.alg.astype(str)).to_dict(orient="records")
            stats[scen]["by_rate"] = s_rate.assign(alg=s_rate.alg.astype(str)).to_dict(orient="records")
            stats[scen]["by_imputer_overall"] = pairs.groupby("imputer")["delta_r2"].mean().to_dict()
            stats[scen]["by_rate_overall"] = pairs.groupby("perc-miss")["delta_r2"].mean().to_dict()
            rate_corr = pairs.groupby(["ds", "alg", "imputer", "perc-miss"])["delta_r2"].mean().reset_index()
            stats[scen]["rate_spearman"] = dict(zip(["rho", "p"], spearmanr(rate_corr["perc-miss"],
                                                                         rate_corr["delta_r2"])))
            fig_imp(pairs)

    fig_overview_box(all_pairs)
    stats["size_corr_fig"] = fig_size(all_pairs)

    with open(os.path.join(TCC, "scripts", "estatisticas.json"), "w") as f:
        json.dump(stats, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("  salvo: tcc/scripts/estatisticas.json")


if __name__ == "__main__":
    main()
