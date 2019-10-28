
from mosaic_ml.model_config.util import check_for_bool, check_none


class LibLinear_Preprocessor:
    # Liblinear is not deterministic as it uses a RNG inside
    def __init__(self, penalty, loss, dual, tol, C, multi_class,
                 fit_intercept, intercept_scaling, class_weight=None,
                 random_state=None):
        self.penalty = penalty
        self.loss = loss
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, Y):
        import sklearn.svm
        from sklearn.feature_selection import SelectFromModel

        self.C = float(self.C)
        self.tol = float(self.tol)
        self.dual = check_for_bool(self.dual)
        self.fit_intercept = check_for_bool(self.fit_intercept)
        self.intercept_scaling = float(self.intercept_scaling)

        if check_none(self.class_weight):
            self.class_weight = None

        estimator = sklearn.svm.LinearSVC(penalty=self.penalty,
                                          loss=self.loss,
                                          dual=self.dual,
                                          tol=self.tol,
                                          C=self.C,
                                          class_weight=self.class_weight,
                                          fit_intercept=self.fit_intercept,
                                          intercept_scaling=self.intercept_scaling,
                                          multi_class=self.multi_class,
                                          random_state=self.random_state)

        estimator.fit(X, Y)
        self.preprocessor = SelectFromModel(estimator=estimator,
                                            threshold='mean',
                                            prefit=True)

        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)


def get_model(name, config, random_state):
    list_param = {"random_state": random_state,
                  "class_weight": "weighting" if config["class_weight"] == "weighting" else None}
    for k in config:
        if k.startswith("preprocessor:liblinear_svc_preprocessor:"):
            param_name = k.split(":")[2]
            list_param[param_name] = config[k]
    model = LibLinear_Preprocessor(**list_param)
    return (name, model)
