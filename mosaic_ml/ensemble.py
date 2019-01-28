import numpy as np
from sklearn import preprocessing


class Ensemble():
    def __init__(self, datamanager, nb_ensemble, nb_best, scoring_func):
        self.datamanager = datamanager
        self.nb_ensemble = nb_ensemble
        self.nb_best = nb_best
        self.scoring_func = scoring_func

    def add_data(self, y_valid, y_test, score_valid):
        # return true if data is added
        return self.datamanager.add_data(y_valid, y_test, score_valid)

    def fit_label(self, y_train):
        self.labelizer = preprocessing.LabelBinarizer()
        self.labelizer.fit(y_train)

    def fit(self):
        self.ens_index = set()
        self.ens_set = np.zeros(len(self.datamanager.y_valid))
        for i in range(self.nb_ensemble):
            index = np.argmax([
                self.performance(self.ens_set, new_el, i) for new_el in self.datamanager.valid_preds
            ])
            self.ens_index.add(index)
            self.ens_set  = self.ens_set + self.datamanager.valid_preds[index]

    def predict(self):
        np.mean([self.datamanager.test_preds[i] in i in self.ens_index], 0)


    def performance(ens_set, new_el, size_ens_set):
        return self.scoring_func(np.argmax((ens_set + self.labelizer.transform(new_el)) / (size_ens_set + 1)), self.datamanager.y_valid)

    def predict(self):
        pass
