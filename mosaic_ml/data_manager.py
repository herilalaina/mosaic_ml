import shutil
import os, glob
import pickle
import numpy as np

class DataManager():
    def __init__(self, name_directory, nb_ensemble=20):
        self.dirpath = os.path.join(name_directory)
        if os.path.exists(self.dirpath) and os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)
        os.makedirs(self.dirpath)

        self.nb_ensemble = nb_ensemble
        self.nb_models_in_batch = [0] * 10
        self.list_score = []

        self.current_batch = 0


    def init_batch(self, batch, X_train, y_train):
        self.current_batch = batch
        dir_batch = os.path.join(self.dirpath, str(batch))
        if not os.path.exists(dir_batch):
            os.makedirs(dir_batch)
            pickle.dump(X_train, open(os.path.join(dir_batch, "X_train.p"), "wb"))
            pickle.dump(y_train, open(os.path.join(dir_batch, "y_train.p"), "wb"))

    def add_data(self, score, models):
        dir_batch = os.path.join(self.dirpath, str(self.current_batch))
        self.nb_models = len(models)
        if len(self.list_score) < self.nb_ensemble:
            self.list_score.append(score)
            index_new = len(self.list_score) - 1
            for i, model in enumerate(models):
                pickle.dump(model,
                            open(os.path.join(dir_batch, "model_{0}_{1}.p".format(index_new, i)), "wb"))
        else:
            index_new = np.argmin(self.list_score)
            for i, model in enumerate(models):
                pickle.dump(model,
                            open(os.path.join(dir_batch, "model_{0}_{1}.p".format(index_new, i)), "wb"))

    def get_X_y(self, batch):
        dir_batch = os.path.join(self.dirpath, str(batch))
        if os.path.exists(dir_batch):
            return (pickle.load(open(os.path.join(dir_batch, "X_train.p"), "rb")),
                    pickle.load(open(os.path.join(dir_batch, "y_train.p"), "rb")))
        return None

    def _get_model(self, batch, index):
        dir_batch = os.path.join(self.dirpath, str(batch))
        if os.path.exists(dir_batch):
            list_model = []
            for i, model in range(self.nb_models):
                list_model.append(pickle.load(open(os.path.join(dir_batch, "model_{0}_{1}.p".format(index, i)), "rb")))
            return list_model

    def get_models(self, batch):
        all_precedent_model = []
        for b in range(batch + 1):
            nb = self.get_nb_model(b)
            if nb > 0:
                for i in range(nb):
                    all_precedent_model.extend(self._get_model(b, i))
        return all_precedent_model

    def get_nb_model(self, batch):
        dir_batch = os.path.join(self.dirpath, str(batch))
        nb = 0
        for f in glob.glob(dir_batch):
            if "model_" in f:
                nb += 1
        return nb

    def __exit__(self, exc_type, exc_value, traceback):
        if os.path.exists(self.dirpath) and os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)
