from examples.update_metadata_util import load_task, classification_tasks
from mosaic_ml.automl import AutoML


def test_get_metalearning_AS():
    from mosaic_ml.autosklearn_wrapper.autosklearn import get_autosklearn_metalearning
    for task in classification_tasks[::10]:
        X_train, y_train, X_test, y_test, cat = load_task(task)
        initial_configuration_metalearning_AS, list_dataset_NN = get_autosklearn_metalearning(X_train, y_train, cat, "balanced_accuracy", 50)
        assert len(initial_configuration_metalearning_AS) == len(list_dataset_NN)
        assert sum([config is not None for config in initial_configuration_metalearning_AS]) == len(initial_configuration_metalearning_AS)
        assert sum([dataset is not None for dataset in list_dataset_NN]) == len(list_dataset_NN)

def test_metalearning():
    from mosaic_ml.autosklearn_wrapper.autosklearn import get_autosklearn_metalearning

    for task in [252, 9971]:
        X_train, y_train, X_test, y_test, cat = load_task(task)

        autoML = AutoML(time_budget=60,
                        time_limit_for_evaluation=30,
                        memory_limit=3024,
                        seed=1,
                        scoring_func="balanced_accuracy",
                        verbose=0,
                        ensemble_size=0
                        )

        intial_configuration_metalearning_AS, _ = get_autosklearn_metalearning(X_train, y_train, cat, "balanced_accuracy", 50)

        best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test,
                                                categorical_features=cat,
                                                initial_configurations=intial_configuration_metalearning_AS[:30]) # init with 30 configs
        assert best_config is not None
        assert isinstance(best_score, float)
