from autosklearn.pipeline.components.classification.qda import QDA

def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:qda:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = QDA(**list_param)
    return (name, model)
