from examples.update_metadata_util import load_task, classification_tasks
from mosaic_ml.automl import AutoML


def test_vanilla():
    for task in [252, 9971]:
        X_train, y_train, X_test, y_test, cat = load_task(task)

        autoML = AutoML(time_budget=60,
                        time_limit_for_evaluation=30,
                        memory_limit=3024,
                        seed=1,
                        scoring_func="balanced_accuracy",
                        verbose=0
                        )

        best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
        assert best_config is not None
        assert isinstance(best_score, float)
