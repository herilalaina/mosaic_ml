import argparse
import sys

sys.path.insert(0, '/home/tau/hrakotoa/Code/mosaic_project/mosaic_ml')
sys.path.append('.')

from update_metadata_util import load_task

import pickle
import pynisher
import scipy

from mosaic_ml.utils import balanced_accuracy

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

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

info = {
    "working_directory": tmp_dir + '/',
    "images_directory": tmp_dir + "/images"
}

X_train, y_train, X_test, y_test, cat = load_task(task_id)
imp = Imputer(missing_values="NaN", strategy="median")
imp.fit(X_train)

X_train = imp.transform(X_train)
X_test = imp.transform(X_test)

if not scipy.sparse.issparse(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
else:
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


def train_predict_func(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    return balanced_accuracy(y_valid, pipeline.predict(X_valid))


os.chdir(info["working_directory"])
list_score = []
error_log = []
for file in sorted(glob.glob('*.pkl'), key=os.path.getmtime):
    pipeline = pickle.load(open(file, "rb"))
    try:
        searcher = pynisher.enforce_limits(mem_in_mb=3072 * 2)(train_predict_func)
        score = searcher(pipeline, X_train, y_train, X_test, y_test)
        pipeline.fit(X_train, y_train)
        list_score.append((file, score))
    except Exception as e:
        list_score.append((file, 0))
        error_log.append((file, e))

with open("{0}/validation.txt".format(tmp_dir), "w") as f:
    for file, score in list_score:
        f.write("{0},{1}\n".format(file, score))
print("Validation created for task={0} run={1}".format(task_id, seed))

for file, error in error_log:
    with open("{0}/error.txt".format(tmp_dir), "a+") as f:
        f.write("{0},{1}\n".format(file, error))
