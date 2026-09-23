import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do projeto (onde está model_utils.py)

import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import train_test_split

from model_utils import (
    NUM_TRIALS, datasets, target_algorithms, fetch_data,
    build_model, multiple_score, limit_rows,
    load_results, already_done, append_and_save, timer, output_path,
)

OUTPUT = output_path("feature_selection_results.csv")
PERCENTILES = [1, 5, 10, 20]


def run_feature_selection_test():
    print("Feature Selection")
    print("Datasets to be used: ", datasets, flush=True)

    results = load_results(OUTPUT)

    for ds in datasets:
        print("\n TESTING ds: _", ds, "_\n", flush=True)
        df_full = fetch_data(ds)

        for alg in target_algorithms:
            print("alg: ", alg, flush=True)
            df = limit_rows(df_full, alg)
            X = df.drop('target', axis=1)
            y = df['target']
            n_features = X.shape[1]

            for perc in PERCENTILES:
                print("perc: ", perc, flush=True)
                if already_done(results, ds=ds, alg=alg, perc=perc):
                    continue

                # Garante pelo menos 1 feature — e usa o MESMO percentil nos dois cenários
                # (antes só o cenário com vazamento usava o percentil ajustado).
                actual_perc = perc
                if n_features * (perc / 100) < 1:
                    actual_perc = 100 / n_features
                    print(f"  perc {perc} resulta em <1 feature em {ds}; usando {actual_perc:.2f}", flush=True)

                # COM vazamento: a seleção enxerga todos os dados (inclusive o futuro conjunto de teste).
                leak_selector = SelectPercentile(score_func=f_regression, percentile=actual_perc).fit(X, y)
                X_leak = leak_selector.transform(X)

                rows = []
                with timer(f"{ds}/{alg}/perc={perc}"):
                    for i in range(NUM_TRIALS):
                        idx_train, idx_test = train_test_split(
                            np.arange(len(X)), test_size=0.2, random_state=i)
                        y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

                        # SEM vazamento: a seleção é ajustada só no treino (dentro do pipeline).
                        noleak_model = build_model(
                            alg, selector=SelectPercentile(score_func=f_regression, percentile=actual_perc))
                        noleak_model.fit(X.iloc[idx_train], y_train)
                        noleak = multiple_score(y_test, noleak_model.predict(X.iloc[idx_test]))

                        # COM vazamento: features já escolhidas com todos os dados.
                        leak_model = build_model(alg)
                        leak_model.fit(X_leak[idx_train], y_train)
                        leak = multiple_score(y_test, leak_model.predict(X_leak[idx_test]))

                        rows.append({**leak, "ds": ds, "iteration": i, "alg": alg, "leak": True, "perc": perc})
                        rows.append({**noleak, "ds": ds, "iteration": i, "alg": alg, "leak": False, "perc": perc})
                        print(f"  trial {i}: mse leak={leak['test_mse']:.6g} | noleak={noleak['test_mse']:.6g}",
                              flush=True)

                results = append_and_save(results, rows, OUTPUT)
