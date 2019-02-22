from autosklearn.pipeline.components.classification.sgd import SGD


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:sgd:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = SGD(**list_param)
    return (name, model)
