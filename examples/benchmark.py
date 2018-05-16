import argparse
import json
import logging
import os
import sys
sys.path.insert(0,'/home/herilalaina/Code/software/mosaic_ml')
sys.path.append('.')


from update_metadata_util import load_task

import warnings
import pickle
import pynisher

from sklearn import datasets, feature_selection, linear_model, feature_selection
from sklearn.model_selection import train_test_split

from mosaic_ml.automl import AutoML
from mosaic_ml.utils import balanced_accuracy

import glob, os

parser = argparse.ArgumentParser()
parser.add_argument('--working-directory', type=str, required=True)
parser.add_argument('--time-limit', type=int, required=True)
parser.add_argument('--per-run-time-limit', type=int, required=True)
parser.add_argument('--task-id', type=int, required=True)
parser.add_argument('--task-type', choices=['classification', 'regression'],
                    required=True)
parser.add_argument('-s', '--seed', type=int, required=True)
parser.add_argument('--unittest', action='store_true')
args = parser.parse_args()

working_directory = args.working_directory
time_limit = args.time_limit
per_run_time_limit = args.per_run_time_limit
task_id = args.task_id
task_type = args.task_type
seed = args.seed
is_test = args.unittest


tmp_dir = os.path.join(working_directory, str(task_id))
try:
    os.makedirs(working_directory)
    os.makedirs(tmp_dir)
except:
    pass


info = {
    "working_directory": tmp_dir + '/'
}


X_train, y_train, X_test, y_test, cat = load_task(task_id)

autoML = AutoML(training_log_file = "{0}/result.txt".format(tmp_dir), info_training = info)
autoML.fit(X_train, y_train)

os.chdir(info["working_directory"])
for file in glob.glob("*.pkl"):
    pipeline = pickle.load(open(file, "rb"))
    pipeline.fit(X_train, y_train)
    score = balanced_accuracy(y_test, pipeline.predict(X_test))
    with open("{0}/validation.txt".format(tmp_dir), "a+") as f:
        f.write("{0},{1}\n".format(file, score))
