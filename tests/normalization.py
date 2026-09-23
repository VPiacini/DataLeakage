import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do projeto (onde está model_utils.py)

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler

from model_utils import (
    N_JOBS, NUM_TRIALS, datasets, target_algorithms, fetch_data,
    build_model, get_scoring, make_dict_mean, limit_rows,
    load_results, already_done, append_and_save, timer, output_path,
)

OUTPUT = output_path("normalization_results.csv")


def run_normalization_test():
    print("Normalization")
    print("Datasets to be used: ", datasets, flush=True)

    results = load_results(OUTPUT)
    scoring = get_scoring()

    for ds in datasets:
        print("\n TESTING ds: _", ds, "_\n", flush=True)
        df_full = fetch_data(ds)

        for alg in target_algorithms:
            print("alg: ", alg, flush=True)
            if already_done(results, ds=ds, alg=alg):
                continue

            df = limit_rows(df_full, alg)
            X = df.drop('target', axis=1)
            y = df['target']

            # COM vazamento: o scaler enxerga o dataset inteiro (antes do CV).
            # O resultado não depende do trial, então é calculado uma única vez.
            X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

            rows = []
            with timer(f"{ds}/{alg}"):
                for i in range(NUM_TRIALS):
                    cv = KFold(n_splits=4, shuffle=True, random_state=i)

                    # SEM vazamento: o scaler é ajustado só no treino de cada fold (dentro do pipeline).
                    noleak = make_dict_mean(cross_validate(
                        build_model(alg), X, y, cv=cv, scoring=scoring, n_jobs=N_JOBS))

                    # COM vazamento: dados já escalados, pipeline sem scaler.
                    leak = make_dict_mean(cross_validate(
                        build_model(alg, scale_x=False), X_scaled, y, cv=cv, scoring=scoring, n_jobs=N_JOBS))

                    rows.append({**leak, "iteration": i, "alg": alg, "leak": True, "ds": ds})
                    rows.append({**noleak, "iteration": i, "alg": alg, "leak": False, "ds": ds})
                    print(f"  trial {i}: mse leak={leak['test_mse']:.6g} | noleak={noleak['test_mse']:.6g}", flush=True)

            results = append_and_save(results, rows, OUTPUT)
