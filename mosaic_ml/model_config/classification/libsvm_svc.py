from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:libsvm_svc:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = LibSVM_SVC(**list_param)
    return (name, model)
