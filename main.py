from utils.data_loader import *
from utils.preprocessing import *
from utils.trainer import *

X_train, y_train, X_test, y_test = load_cifar10()
X_train, X_test = normalize_images(X_train, X_test)
train_and_evaluate(X_train, y_train, X_test, y_test, max_depth=10, min_samples_split=5)
