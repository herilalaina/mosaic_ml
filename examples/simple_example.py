from mosaic_ml.automl import AutoML
from update_metadata_util import load_task
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


X_train, y_train, X_test, y_test, cat = load_task(6)

autoML = AutoML(time_budget=360,
                time_limit_for_evaluation=100,
                memory_limit=3024,
                seed=1,
                scoring_func="balanced_accuracy",
                exec_dir="execution_dir")

res = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
print(res)
