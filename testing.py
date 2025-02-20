from utils.imports import *
from utils.trainer import *
from utils.data_loader import *
from utils.preprocessing import *

X_train, y_train, X_test, y_test = load_cifar10()
X_train, X_test = normalize_images(X_train, X_test)

max_depth_values = [10, 20, 30, None]  # None significa sin límite
min_samples_split_values = [2, 5, 10]
criterion_values = ["gini", "entropy"]

param_combinations = list(itertools.product(max_depth_values, min_samples_split_values, criterion_values))

log_file = "log_tests.txt"

if os.path.exists(log_file):
    os.remove(log_file)

best_accuracy = 0
best_params = None

with open(log_file, "w") as f:
    for max_depth, min_samples_split, criterion in param_combinations:
        print(f"Probando: max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion}")
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, max_depth, min_samples_split)

        f.write(
            f"max_depth={max_depth}, min_samples_split={min_samples_split}, criterion={criterion} -> Accuracy: {accuracy:.4f}\n")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (max_depth, min_samples_split, criterion)

    f.write(
        f"\nMejor resultado: max_depth={best_params[0]}, min_samples_split={best_params[1]}, criterion={best_params[2]} -> Accuracy: {best_accuracy:.4f}\n")

print("\nPruebas completadas. Revisa el archivo 'log_tests.txt' para ver los resultados.")
