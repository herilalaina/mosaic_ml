from update_metadata_util import load_task

from mosaic_ml.automl import AutoML

X_train, y_train, X_test, y_test, cat = load_task(6)

autoML = AutoML(time_budget=360,
                time_limit_for_evaluation=100,
                memory_limit=3024,
                multi_fidelity=False,
                use_parameter_importance=False,
                use_rave=False,
                seed=1,
                scoring_func="balanced_accuracy")
res = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
print(res)
