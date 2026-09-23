import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do projeto (onde está model_utils.py)

import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from model_utils import (
    N_JOBS, NUM_TRIALS, datasets, target_algorithms, fetch_data,
    build_model, get_scoring, get_imputer, insert_missing_in_df, make_dict_mean, limit_rows,
    load_results, already_done, append_and_save, timer, output_path,
)

OUTPUT = output_path("imputation_results.csv")
PERC_MISS = [0.05, 0.1, 0.2, 0.3]
IMPUTERS = ['mean', 'median', 'KNN']  # , 'iterative'


def run_value_imputation_test():
    print("Value Imputation")
    print("Datasets to be used: ", datasets, flush=True)

    results = load_results(OUTPUT)
    scoring = get_scoring()

    for ds in datasets:
        print("\n TESTING ds: _", ds, "_\n", flush=True)
        df_orig = fetch_data(ds)

        for p in PERC_MISS:
            print("p miss: ", p, flush=True)
            df_full = insert_missing_in_df(df_orig, p)

            for imp in IMPUTERS:
                # COM vazamento: a imputação usa todas as linhas. Depende só de (ds, p, imp),
                # não do algoritmo nem do trial — então é feita uma vez e reaproveitada
                # (o KNNImputer era refeito 5 algoritmos x 6 trials).
                leaky_cache = {}

                def leaky_imputed(df_alg):
                    key = len(df_alg)  # o subconjunto do SVR é determinístico, então o tamanho identifica
                    if key not in leaky_cache:
                        feats = df_alg.drop('target', axis=1)
                        imputed = get_imputer(imp).fit_transform(feats)
                        leaky_cache[key] = pd.DataFrame(imputed, columns=feats.columns)
                    return leaky_cache[key]

                for alg in target_algorithms:
                    print("alg: ", alg, "| imputer:", imp, flush=True)
                    if already_done(results, ds=ds, alg=alg, imputer=imp, **{"perc-miss": p}):
                        continue

                    df = limit_rows(df_full, alg)
                    X = df.drop('target', axis=1)
                    y = df['target']

                    with timer(f"{ds}/p={p}/{imp}/{alg}"):
                        X_leak = leaky_imputed(df)

                        rows = []
                        for i in range(NUM_TRIALS):
                            cv = KFold(n_splits=4, shuffle=True, random_state=i)

                            # SEM vazamento: o imputer é ajustado só no treino de cada fold.
                            noleak = make_dict_mean(cross_validate(
                                build_model(alg, imputer=get_imputer(imp)), X, y,
                                cv=cv, scoring=scoring, n_jobs=N_JOBS))

                            # COM vazamento: dados já imputados; scaler continua dentro do CV.
                            leak = make_dict_mean(cross_validate(
                                build_model(alg), X_leak, y,
                                cv=cv, scoring=scoring, n_jobs=N_JOBS))

                            common = {"ds": ds, "iteration": i, "alg": alg}
                            rows.append({**leak, **common, "leak": True, "imputer": imp, "perc-miss": p})
                            rows.append({**noleak, **common, "leak": False, "imputer": imp, "perc-miss": p})
                            print(f"  trial {i}: mse leak={leak['test_mse']:.6g} | noleak={noleak['test_mse']:.6g}",
                                  flush=True)

                    results = append_and_save(results, rows, OUTPUT)
