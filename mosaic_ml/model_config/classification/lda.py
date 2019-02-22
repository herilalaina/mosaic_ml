from autosklearn.pipeline.components.classification.lda import LDA


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:lda:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = LDA(**list_param)
    return (name, model)
