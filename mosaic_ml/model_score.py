from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pickle, os


class ScoreModel():
    def __init__(self, nb_param, X=None, y=None, id_most_import_class = None, dataset_features = []):
        self.model = RandomForestRegressor()
        self.model_of_time = RandomForestRegressor()
        self.model_general = RandomForestRegressor()
        self.nb_param = nb_param
        self.id_most_import_class = id_most_import_class
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data_score.p")
        self.dataset_features = dataset_features

        self.active_general_model = False

        if X is not None and y is not None:
            self.X = X
            self.y = y
            self.model.fit(X, y)
        else:
            X, y, y_time = self.load_data()
            self.X, self.y, self.y_time = X, y, y_time

        self.nb_added = 0

    def learn_general_model(self, X, y):
        self.model_general.fit(X, y)

    def _normalize_x_with_features(self, x):
        if len(self.dataset_features) > 0:
            return x + self.dataset_features
        return x

    def _normalize_X_with_features(self, X):
        if len(self.dataset_features) > 0:
            return [self._normalize_x_with_features(x) for x in X]
        return X

    def feed_model(self, X, y, y_time):
        self.X, self.y, self.y_time = X, y, y_time
        self.fit()

    def get_performance(self, x):
        output = {}
        try:
            list_pred = []
            for estimator in self.model.estimators_:
                x_pred = estimator.predict([x])
                list_pred.append(x_pred[0])
            output = {"perf_mean": np.mean(list_pred), "perf_std": np.std(list_pred)}
        except Exception as e:
            raise(e)
            output = {"mean": 0, "std": 0}

        """try:
            list_pred = []
            for estimator in self.model_of_time.estimators_:
                x_pred = estimator.predict([x])
                list_pred.append(x_pred[0])
            output["mean_runtime"] = np.mean(list_pred)
            output["std_runtime"] = np.std(list_pred)
        except Exception as e:
            print(e)
            output["mean_runtime"] = 0
            output["std_runtime"] = 0"""

        return output

    def get_mu_sigma_from_rf(self, X, model = "normal"):
        list_pred = []
        if model == "local":
            for estimator in self.model.estimators_:
                x_pred = self.model.predict(X)
                list_pred.append(x_pred)
        elif model == "general":
            try:
                for estimator in self.model_general.estimators_:
                    x_pred = self.model_general.predict([np.concatenate([x, self.dataset_features]) for x in X])
                    list_pred.append(x_pred)
            except Exception as e:
                return np.zeros(len(X)), np.zeros(len(X))

        elif model == "time":
            for estimator in self.model_of_time.estimators_:
                x_pred = self.model_of_time.predict(X)
                list_pred.append(x_pred)
        else:
            raise Exception("Unknown model", model)
        return np.mean(list_pred, axis=0), np.std(list_pred, axis=0)

    def load_data(self):
        try:
            return pickle.load(open(self.path, "rb"))
        except:
            return [], [], []

    def save_data(self, file_dir):
        np.save(os.path.join(file_dir, "X.npy"), self.X)
        np.save(os.path.join(file_dir, "y.npy"), self.y)
        np.save(os.path.join(file_dir, "y_time.npy"), self.y_time)

    def partial_fit(self, x, y, y_time):
        if y > 0:
            self.X.append(x)
            self.y.append(y)
            self.y_time.append(y_time)
            self.fit()
            self.nb_added += 1

    def fit(self):
        sample_weight = self._get_sample_weight()
        self.model.fit(self.X, self.y, sample_weight=sample_weight)
        #self.model_of_time.fit(self.X, self.y_time, sample_weight=sample_weight)
        self.model_general.fit([np.concatenate([x, self.dataset_features])  for x in self.X], self.y)

    def importance_variable(self):
        if check_is_fitted(self.model):
            return self.model.feature_importances_
        else:
            raise NotFittedError("ScoreModel not fitted")

    def predict(self, x):
        if check_is_fitted(self.model):
            return self.model.predict([x])[0]
        else:
            raise NotFittedError("ScoreModel not fitted")

    def most_importance_parameter(self, ids):
        if self.nb_added > 5:
            weights = [np.abs(self.model.feature_importances_[id - self.nb_param]) for id in ids]
            weights = weights / sum(weights)
            return np.random.choice(list(range(len(ids))), p=weights)
        else:
            return np.random.randint(len(ids))

    def _get_sample_weight(self):
        count_id = {}
        for x in self.X:
            x_ = tuple(x[i] for i in self.id_most_import_class)
            if x_ in count_id:
                count_id[x_] = count_id[x_] + 1
            else:
                count_id[x_] = 1

        sample_weight = []
        for x in self.X:
            x_ = tuple(x[i] for i in self.id_most_import_class)
            sample_weight.append((1.0 / count_id[x_]))

        return sample_weight

    def rave_value(self, value, idx, is_categorical, range_value):
        #print(value)
        if len(value) == 1:
            return value[0]
        elif(len(self.X) < 10):
            return np.random.choice(value)

        #TODO: to optimize
        N = len(self.X)
        X_ = np.array(self.X)
        Y_ = np.array(self.y)

        if is_categorical:
            list_value = [0] * len(range_value)

            for v in range(len(range_value)):
                if list_value[v] != 0:
                    list_score = Y_[X_[:, idx - self.nb_param] == v]
                    if len(list_value) > 0:
                        list_value[v] = np.mean(list_value) + np.sqrt(2 * np.log10(N) / len(list_value))
                    else:
                        list_value[v] = 10

            id_max = np.argmax([list_value[v] for v in value])
            return value[id_max]
        else:
            list_value = [0] * len(value)
            res = (X_[:, idx] != 0)
            X = X_[res, :]
            Y = Y_[res]

            if len(Y) > 10:
                sigma = np.std(X[:, idx])
                for i, v in enumerate(value):
                    T = np.sum([np.exp(-(v - x[idx]) ** 2 / (2 * sigma ** 2)) for x, y in zip(X, Y)])
                    if T != 0:
                        list_value[i] = np.sum([np.exp(-(v - x[idx]) ** 2 / (2 * sigma ** 2)) * y / T for x, y in zip(X, Y)])
                    else:
                        list_value[i] = 0
            else:
                return np.random.choice(value)
            id_max = np.argmax(list_value)
            return value[id_max]
