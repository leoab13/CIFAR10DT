from utils.imports import *
from utils.model import get_decision_tree_model

def train_and_evaluate(X_train, y_train, X_test, y_test, max_depth=None, min_samples_split=2):
    model = get_decision_tree_model(max_depth, min_samples_split)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Precisión del modelo: {accuracy:.4f}")
    return accuracy
