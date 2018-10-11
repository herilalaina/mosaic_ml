import sys
sys.path.insert(0,'/home/herilalaina/Code/software/mosaic_ml')
import pickle


from update_metadata_util import load_task

from sklearn import datasets, feature_selection, linear_model, feature_selection
from sklearn.model_selection import train_test_split

from mosaic_ml.automl import AutoML
from mosaic_ml.utils import balanced_accuracy

import glob, os

info = {
    "working_directory": "tmp/",
    "images_directory": "examples/img"
}

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

autoML = AutoML(training_log_file = "tmp/result.txt", info_training = info)
autoML.fit(X_train, y_train)

os.chdir(info["working_directory"])
for file in glob.glob("*.pkl"):
    pipeline = pickle.load(open(file, "rb"))
    pipeline.fit(X_train, y_train)
    score = balanced_accuracy(y_test, pipeline.predict(X_test))
    with open("validation.txt", "a+") as f:
        f.write("{0},{1}\n".format(file, score))
