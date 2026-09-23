from tests.normalization import run_normalization_test
from tests.feature_selection import run_feature_selection_test
from tests.tuning import run_hyperparameter_tuning_test
from tests.imputation import run_value_imputation_test

TESTS = {
    "normalization": run_normalization_test,
    "feature_selection": run_feature_selection_test,
    "tuning": run_hyperparameter_tuning_test,
    "imputation": run_value_imputation_test,
}

def run_tests(test_type):
    test = TESTS.get(test_type)
    if test is None:
        print(f"Invalid test type: {test_type}")
        print(f"Available tests: {', '.join(TESTS.keys())}")
        return

    test()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python main.py <test_type>")
        print(f"Where <test_type> is one of: {', '.join(TESTS.keys())}")
        sys.exit(1)

    run_tests(sys.argv[1])