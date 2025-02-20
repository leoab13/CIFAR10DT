from utils.imports import *

def normalize_images(X_train, X_test):
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return X_train, X_test
