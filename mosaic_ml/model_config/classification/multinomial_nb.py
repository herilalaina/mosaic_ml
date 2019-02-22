from autosklearn.pipeline.components.classification.multinomial_nb import MultinomialNB


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("classifier:multinomial_nb:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = MultinomialNB(**list_param)
    return (name, model)
