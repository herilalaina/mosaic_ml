from autosklearn.pipeline.components.feature_preprocessing.kernel_pca import KernelPCA


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:kernel_pca:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = KernelPCA(**list_param)
    return (name, model)
