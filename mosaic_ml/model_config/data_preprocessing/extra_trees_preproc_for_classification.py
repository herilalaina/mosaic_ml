from autosklearn.pipeline.components.feature_preprocessing.extra_trees_preproc_for_classification import ExtraTreesPreprocessorClassification


def get_model(name, config, random_state):
    list_param = {"random_state": random_state}
    for k in config:
        if k.startswith("preprocessor:extra_trees_preproc_for_classification:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = ExtraTreesPreprocessorClassification(**list_param)
    return (name, model)
