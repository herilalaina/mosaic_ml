from mosaic_ml.model_config.classification import get_classifier
from mosaic_ml.model_config.data_preprocessing import get_data_preprocessing

from pynisher import TimeoutException, MemorylimitException


def evaluate_imputation(imputation_strategy):
    from sklearn.preprocessing import Imputer

    imp = Imputer(strategy=imputation_strategy)
    return ("Imputation", imp)


def evaluate_encoding(choice, config, categorical_features, is_sparse):
    from mosaic_ml.model_config.encoding.OneHotEncoding import OneHotEncoder
    from sklearn.preprocessing import FunctionTransformer

    if choice == "no_encoding":
        encoding = FunctionTransformer()
    elif choice == "one_hot_encoding":
        if config["use_minimum_fraction"]:
            minimum_fraction = config["minimum_fraction"]
        else:
            minimum_fraction = None

        if categorical_features is None:
            categorical_features = []
        encoding = OneHotEncoder(categorical_features=categorical_features,
                                 minimum_fraction=minimum_fraction,
                                 sparse=is_sparse)
    else:
        raise NotImplemented("Not implemented {0}".format(choice))

    return ("categorical_encoding", encoding)


def evaluation_rescaling(choice, config):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.preprocessing import Normalizer
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import StandardScaler

    if choice == "minmax":
        scaler = MinMaxScaler()
    elif choice == "none":
        scaler = FunctionTransformer()
    elif choice == "normalize":
        scaler = Normalizer()
    elif choice == "quantile_transformer":
        scaler = QuantileTransformer(n_quantiles=int(config["rescaling:quantile_transformer:n_quantiles"]),
                                     output_distribution=config["rescaling:quantile_transformer:output_distribution"])
    elif choice == "robust_scaler":
        scaler = RobustScaler(quantile_range=(float(config["rescaling:robust_scaler:q_min"]),
                                              float(config["rescaling:robust_scaler:q_max"])))
    elif choice == "standardize":
        scaler = StandardScaler()
    else:
        raise NotImplemented("Scaler not implemented")

    return ("scaler", scaler)


def get_sample_weight(y):
    import numpy as np
    if len(y.shape) > 1:
        offsets = [2 ** i for i in range(y.shape[1])]
        Y_ = np.sum(y * offsets, axis=1)
    else:
        Y_ = y

    unique, counts = np.unique(Y_, return_counts=True)
    cw = 1. / counts
    cw = cw / np.mean(cw)

    sample_weights = np.ones(Y_.shape)

    for i, ue in enumerate(unique):
        mask = Y_ == ue
        sample_weights[mask] *= cw[i]
    return sample_weights


def config_to_pipeline(config, categorical_features, is_sparse):
    from sklearn.pipeline import Pipeline

    list_params = config.keys()

    balancing_strategy = config["balancing:strategy"]
    imputation_strategy = config["imputation:strategy"]
    categorical_encoding__choice__ = config["categorical_encoding:__choice__"]
    rescaling__choice__ = config["rescaling:__choice__"]

    classifier__choice__ = config["classifier:__choice__"]
    preprocessor__choice__ = config["preprocessor:__choice__"]

    name_pre, model_pre = get_data_preprocessing.evaluate(preprocessor__choice__, config)
    name_clf, model_clf = get_classifier.evaluate_classifier(classifier__choice__, config)

    if balancing_strategy == "weighting":
        if name_clf in ['decision_tree', 'extra_trees', 'liblinear_svc',
                        'libsvm_svc', "passive_aggressive", "random_forest"]:
            model_clf.set_params(class_weight='balanced')
        if name_pre in ['liblinear_svc_preprocessor', 'extra_trees_preproc_for_classification']:
            model_pre.estimator.set_params(class_weight='balanced')

    pipeline_list = [
        evaluate_imputation(imputation_strategy),
        evaluate_encoding(categorical_encoding__choice__, config, categorical_features, is_sparse),
        evaluation_rescaling(rescaling__choice__, config),
        (name_pre, model_pre),
        (name_clf, model_clf)
    ]

    pipeline = Pipeline(pipeline_list)
    return pipeline, balancing_strategy == "weighting"


def evaluate(config, bestconfig, X=None, y=None, score_func=None, categorical_features=None, seed=None):
    print("*", end="")
    try:
        from scipy.sparse import issparse
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.model_selection import StratifiedKFold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X))
            list_score = []

            name_clf = pipeline.steps[4][0]

            skf = StratifiedKFold(n_splits=5, random_state=seed)
            for train_index, test_index in skf.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]

                fit_params = {}
                if balancing_strategy and name_clf in ['adaboost', 'gradient_boosting', 'random_forest', 'extra_trees',
                                                       'sgd', 'xgradient_boosting']:
                    fit_params[name_clf + "__sample_weight"] = get_sample_weight(y_train)

                pipeline.fit(X_train, y_train, **fit_params)
                list_score.append(score_func(y_test, pipeline.predict(X_test)))

                #if list_score[-1] < bestconfig["score_validation"]:
                #    return {"validation_score": list_score[-1]}

            return {"validation_score": sum(list_score) / len(list_score)}
    except TimeoutException as e:
        raise(e)
    except MemorylimitException as e:
        raise(e)
    except Exception as e:
        print(config)
        raise (e)


def test_function(config, X_train, y_train, X_test, y_test, categorical_features=None):
    from scipy.sparse import issparse
    from sklearn.metrics import balanced_accuracy_score
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    with warnings.catch_warnings():
        pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X_train))
        fit_params = {}
        name_clf = pipeline.steps[4][0]
        if balancing_strategy and name_clf in ['adaboost', 'gradient_boosting', 'random_forest', 'extra_trees', 'sgd',
                                               'xgradient_boosting']:
            fit_params[name_clf + "__sample_weight"] = get_sample_weight(y_train)
        pipeline.fit(X_train, y_train, **fit_params)
        y_pred = pipeline.predict(X_test)
        return balanced_accuracy_score(y_pred, y_test)
