from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:extra_trees:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = ExtraTreesClassifier(**list_param)
    return (name, model)
