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
        self.last_validation = 0

    def _get_data(self, id):
        valids_files = []
        test_files = []
        fetched_files = []
        for run in self.runhistory:
            if run["validation_score"] > 0:
                fetched_files.append((run["id"], run["validation_score"], run["model"]))

            if run["id"] >= id:
                break

        valids_dis = sorted(fetched_files, key= lambda i: i[1]) #[:self.nb_best]
        id_ensemble = {}
        for id, s, model in valids_dis:
            id_ensemble[model["classifier:__choice__"]] = (id, s)

        final_id = []
        for name_clf, (id_to_fetch, s) in id_ensemble.items():
            val_file = np.load(os.path.join(self.ensemble_directory, "pred_valid_{0}.npy".format(id_to_fetch)))
            test_file = np.load(os.path.join(self.ensemble_directory, "pred_test_{0}.npy".format(id_to_fetch)))
            valids_files.append(val_file)
            test_files.append(test_file)
            final_id.append(id_to_fetch)
            #self.last_validation = s

        return final_id, valids_files, test_files

    def _merge_pred(self, current_set, id_to_add, valid_files):
        if len(current_set) == 0:
            return valid_files[id_to_add]
        return np.around(np.mean([valid_files[id_] for id_ in current_set] + [valid_files[id_to_add]], 0))

    def _build_ensemble(self, ids, valid_files):
        if len(ids) != len(valid_files):
            raise Exception("Length for ids, valid file is not the same")

        if len(ids) == 1:
            return [0]

        current_set = []
        for _ in range(self.nb_ensemble):
            id_max = np.argmax([self.scoring_func(self.y_valid, self._merge_pred(current_set, id_, valid_files)) for id_ in range(len(ids))])
            current_set.append(id_max)

        return current_set

    def predict_ensemble(self, ensemble_set, test_files, true_label):
        return self.scoring_func(true_label, np.around(np.mean([test_files[id_] for id_ in ensemble_set], 0)))

    def score_ensemble(self, y):
        scores = []

        best_score = {}
        for run in self.runhistory:
            model = run["model"]
            print(".", end=".")
            if run["validation_score"] > 0 and (model["classifier:__choice__"] not in best_score or run["validation_score"] > best_score[model["classifier:__choice__"]]): #self.last_validation:
                valids_dis, valids_files, test_files = self._get_data(run["id"])
                ens_set = self._build_ensemble(valids_dis, valids_files)
                score_valid = self.predict_ensemble(ens_set, valids_files, self.y_valid)
                score_test = self.predict_ensemble(ens_set, test_files, y)
                scores.append([run["elapsed_time"], score_valid, score_test])
                print([run["elapsed_time"], score_valid, score_test])
                best_score[model["classifier:__choice__"]] = run["validation_score"]
        return scores
