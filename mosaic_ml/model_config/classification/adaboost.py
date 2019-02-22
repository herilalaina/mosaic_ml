from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from sklearn.tree import DecisionTreeClassifier


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:adaboost:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = AdaboostClassifier(**list_param)
    return (name, model)
