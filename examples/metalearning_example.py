from mosaic_ml.automl import AutoML
from update_metadata_util import load_task

from mosaic_ml.autosklearn_wrapper.autosklearn import get_autosklearn_metalearning


if __name__=="__main__":
    X_train, y_train, X_test, y_test, cat = load_task(273)

    autoML = AutoML(time_budget=120,
                    time_limit_for_evaluation=100,
                    memory_limit=3024,
                    seed=1,
                    scoring_func="balanced_accuracy",
                    verbose=True
                    )

    intial_configuration_metalearning_AS, _ = get_autosklearn_metalearning(X_train, y_train, cat, "balanced_accuracy", 50)
    best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test,
                                            categorical_features=cat,
                                            initial_configurations=intial_configuration_metalearning_AS[:30]) # init with 30 configs
    print("Best config {0}\t Result:{1}".format(best_config, best_score))
