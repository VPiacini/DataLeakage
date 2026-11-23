import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do projeto (onde está model_utils.py)

from sklearn.model_selection import KFold, GridSearchCV, cross_validate

from model_utils import (
    N_JOBS, NUM_TRIALS, datasets, target_algorithms, fetch_data,
    build_model, param_prefix, get_scoring, make_dict, make_dict_mean,
    load_results, already_done, append_and_save, timer, output_path,
)

OUTPUT = output_path("hyperparameter_tuning_results.csv")

# Grades (nomes de parâmetros do regressor, sem prefixo; o prefixo do pipeline é adicionado abaixo).
# O kernel 'poly' ficou de fora do SVR: com C alto ele é de longe o mais lento.
HYPERPARAMETER_GRIDS = {
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan']},
    "SVR": {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']},
    "LR": {
        'fit_intercept': [True, False],
        'positive': [True, False]},
    "DT": {
        'max_depth': [3, 5, 8, 10, None]},
    "RF": {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 8, 10, None]},
}


def run_hyperparameter_tuning_test():
    print("Hyperparameter Tuning")
    print("Datasets to be used: ", datasets, flush=True)

    results = load_results(OUTPUT)
    scoring = get_scoring()

    for ds in datasets:
        print("\n TESTING ds: _", ds, "_\n", flush=True)
        df = fetch_data(ds)
        # O tuning usa o dataset completo (sem subamostragem), inclusive no SVR com kernel.
        X = df.drop('target', axis=1)
        y = df['target']

        for alg in target_algorithms:
            print("alg: ", alg, flush=True)
            if already_done(results, ds=ds, alg=alg):
                continue

            prefix = param_prefix(alg)
            p_grid = {prefix + name: values for name, values in HYPERPARAMETER_GRIDS[alg].items()}

            rows = []
            with timer(f"{ds}/{alg}"):
                for i in range(NUM_TRIALS):
                    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
                    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

                    # COM vazamento: escolhe os hiperparâmetros e reporta o score do CV
                    # nos mesmos dados (estimativa otimista). refit=False: só cv_results_ importa.
                    search = GridSearchCV(
                        estimator=build_model(alg, for_tuning=True), param_grid=p_grid,
                        cv=outer_cv, scoring=scoring, refit=False, n_jobs=N_JOBS)
                    search.fit(X, y)
                    leak = make_dict(search.cv_results_)

                    # SEM vazamento: CV aninhado (a busca roda só dentro de cada fold externo).
                    nested_search = GridSearchCV(
                        estimator=build_model(alg, for_tuning=True), param_grid=p_grid,
                        cv=inner_cv, scoring=scoring, refit='neg_mean_squared_error', n_jobs=N_JOBS)
                    noleak = make_dict_mean(cross_validate(
                        nested_search, X=X, y=y, cv=outer_cv, scoring=scoring))

                    rows.append({**leak, "ds": ds, "iteration": i, "alg": alg, "leak": True})
                    rows.append({**noleak, "ds": ds, "iteration": i, "alg": alg, "leak": False})
                    print(f"  trial {i}: mse leak={leak['test_mse']:.6g} | noleak={noleak['test_mse']:.6g}",
                          flush=True)

            results = append_and_save(results, rows, OUTPUT)
