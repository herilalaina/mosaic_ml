from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC

def get_model(choice, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:liblinear_svc:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]

    model = LibLinear_SVC(**list_param)
    return (choice, model)
