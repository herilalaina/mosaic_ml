import sys
sys.path.insert(0,'/home/herilalaina/Code/software/mosaic_ml')
sys.path.insert(0,"/home/tau/hrakotoa/Code/reproduce/mosaic_ml")
import pickle


from update_metadata_util import load_task

from sklearn import datasets, feature_selection, linear_model, feature_selection
from sklearn.model_selection import train_test_split

from mosaic_ml.automl import AutoML
from mosaic_ml.utils import balanced_accuracy

X_train, y_train, X_test, y_test, cat = load_task(3)

autoML = AutoML(time_budget = 300,
                 time_limit_for_evaluation = 10,
                 memory_limit = 3024,
                 multi_fidelity=False,
                 use_parameter_importance=False,
                 use_rave=False)
res = autoML.fit(X_train, y_train, X_test, y_test)

print(res)
