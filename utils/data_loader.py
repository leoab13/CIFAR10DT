from utils.imports import *

DATA_PATH = "data/"

def load_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch[b'labels'])
    return X, y

def load_cifar10():
    x_train, y_train = [], []
    for i in range(1, 6):
        file = os.path.join(DATA_PATH, f"data_batch_{i}")
        x, y = load_batch(file)
        x_train.append(x)
        y_train.append(y)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test, y_test = load_batch(os.path.join(DATA_PATH, "test_batch"))
    return x_train, y_train, x_test, y_test