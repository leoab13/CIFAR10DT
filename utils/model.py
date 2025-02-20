from utils.imports import *

def get_decision_tree_model(max_depth=None, min_samples_split=2,criterion="entropy"):
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
    return model
