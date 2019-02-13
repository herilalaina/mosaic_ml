import os
import numpy as np
from sklearn import preprocessing


class Ensemble():
    def __init__(self, runhistory, nb_ensemble, nb_best, scoring_func, exec_dir):
        self.runhistory = runhistory
        self.nb_ensemble = nb_ensemble
        self.nb_best = nb_best
        self.scoring_func = scoring_func
        self.exec_dir = exec_dir
        self.ensemble_directory = os.path.join(exec_dir, "ensemble_files")
        self.y_valid = np.load(os.join.path(self.ensemble_directory, "y_valid.npy"))
        self.y_test = np.load(os.join.path(self.ensemble_directory, "y_test.npy"))

    def _get_data(self, id):
        valids_files = []
        test_files = []
        fetched_files = []
        for run in self.runhistory:
            fetched_files.append((run["id"], run["validation_score"]))

            if run["id"] >= id:
                break

        valids_dis = sorted(fetched_files, key= lambda i: i[1], reverse = True)[:self.nb_best]
        for id_to_fetch in valids_dis:
            val_file = np.load(os.join.path(self.ensemble_directory, "y_valid_{0}.npy".format(id_to_fetch)))
            test_file = np.load(os.join.path(self.ensemble_directory, "y_test_{0}.npy".format(id_to_fetch)))
            valids_dis.append(val_file)
            test_files.append(test_file)

        return valids_dis, test_files

    def fit_ensemble(self):
        self.ens_index = set()
        self.ens_set = np.zeros(len(self.datamanager.y_valid))
        for i in range(self.nb_ensemble):
            index = np.argmax([
                self.performance(self.ens_set, new_el, i) for new_el in self.datamanager.valid_preds
            ])
            self.ens_index.add(index)
            self.ens_set  = self.ens_set + self.datamanager.valid_preds[index]

