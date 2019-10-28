from mosaic_ml.automl import AutoML
from update_metadata_util import load_task


X_train, y_train, X_test, y_test, cat = load_task(6)

autoML = AutoML(time_budget=120,
                time_limit_for_evaluation=100,
                memory_limit=3024,
                seed=1,
                scoring_func="balanced_accuracy",
                exec_dir="execution_dir"
                )

best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
print("Best config {0}\t Result:{1}".format(best_config, best_score))
