import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do projeto (onde está model_utils.py)

import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from model_utils import (
    N_JOBS, NUM_TRIALS, datasets, target_algorithms, fetch_data,
    build_model, get_scoring, get_imputer, insert_missing_in_df, make_dict_mean,
    load_results, already_done, append_and_save, timer, output_path,
)

OUTPUT = output_path("imputation_results.csv")
PERC_MISS = [0.05, 0.1, 0.2, 0.3]
IMPUTERS = ['mean', 'median', 'KNN']


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
            df = insert_missing_in_df(df_orig, p)
            X = df.drop('target', axis=1)
            y = df['target']

            for imp in IMPUTERS:
                # COM vazamento: a imputação usa todas as linhas. Depende só de (ds, p, imp),
                # não do algoritmo nem do trial, então é feita uma vez e reaproveitada.
                X_leak = None

                for alg in target_algorithms:
                    print("alg: ", alg, "| imputer:", imp, flush=True)
                    if already_done(results, ds=ds, alg=alg, imputer=imp, **{"perc-miss": p}):
                        continue

                    with timer(f"{ds}/p={p}/{imp}/{alg}"):
                        if X_leak is None:
                            X_leak = pd.DataFrame(get_imputer(imp).fit_transform(X), columns=X.columns)

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
