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
        self.y_valid = np.load(os.path.join(self.ensemble_directory, "y_valid.npy"))
        self.y_test = np.load(os.path.join(self.ensemble_directory, "y_test.npy"))

    def _get_data(self, id):
        valids_files = []
        test_files = []
        fetched_files = []
        for run in self.runhistory:
            if run["validation_score"] > 0:
                fetched_files.append((run["id"], run["validation_score"]))

            if run["id"] >= id:
                break

        valids_dis = sorted(fetched_files, key= lambda i: i[1], reverse = True)[:self.nb_best]
        for (id_to_fetch, s) in valids_dis:
            val_file = np.load(os.path.join(self.ensemble_directory, "pred_valid_{0}.npy".format(id_to_fetch)))
            test_file = np.load(os.path.join(self.ensemble_directory, "pred_test_{0}.npy".format(id_to_fetch)))
            valids_files.append(val_file)
            test_files.append(test_file)

        return valids_dis, valids_files, test_files

    def _merge_pred(self, current_set, id_to_add, valid_files):
        if len(current_set) == 0:
            return valid_files[id_to_add]
        return np.around(np.mean([valid_files[id_] for id_ in current_set] + [valid_files[id_to_add]], 0))

    def _build_ensemble(self, ids, valid_files):
        if len(ids) != len(valid_files):
            raise Exception("Length for ids, valid file is not the same")

        if len(ids) == 0:
            return ids

        current_set = []
        for _ in range(self.nb_ensemble):
            id_max = np.argmax([self.scoring_func(self.y_valid, self._merge_pred(current_set, id_, valid_files)) for id_ in range(len(ids))])
            current_set.append(id_max)

        return current_set

    def predict_ensemble(self, ensemble_set, test_files, true_label):
        return self.scoring_func(np.around(np.mean([test_files[id_] for id_ in ensemble_set], 0)), true_label)

    def score_ensemble(self, y):
        scores = []
        for run in self.runhistory:
            print(".", end=".")
            if run["validation_score"] > 0:
                valids_dis, valids_files, test_files = self._get_data(run["id"])
                ens_set = self._build_ensemble(valids_dis, valids_files)
                score_valid = self.predict_ensemble(ens_set, valids_files, self.y_valid)
                score_test = self.predict_ensemble(ens_set, test_files, y)
                scores.append([run["elapsed_time"], score_valid, score_test])
        return scores
