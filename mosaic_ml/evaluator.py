from mosaic_ml.model_config.classification import get_classifier
from mosaic_ml.model_config.data_preprocessing import get_data_preprocessing

from pynisher import TimeoutException, MemorylimitException


def evaluate_imputation(imputation_strategy):
    #from sklearn.preprocessing import Imputer
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy=imputation_strategy, copy=False)
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


def config_to_pipeline(config, type_features, is_sparse, random_state):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    numerical_features = [i for i, x in enumerate(type_features) if x != "categorical"]
    categorical_features = [i for i, x in enumerate(type_features) if x == "categorical"]
    print("numerical_features", numerical_features)
    print("categorical_features", categorical_features)


    list_params = config.keys()

    balancing_strategy = config["balancing:strategy"]
    imputation_strategy = config["imputation:strategy"]
    categorical_encoding__choice__ = config["categorical_encoding:__choice__"]
    rescaling__choice__ = config["rescaling:__choice__"]

    classifier__choice__ = config["classifier:__choice__"]
    preprocessor__choice__ = config["preprocessor:__choice__"]

    name_pre, model_pre = get_data_preprocessing.evaluate(preprocessor__choice__, config, random_state)
    name_clf, model_clf = get_classifier.evaluate_classifier(classifier__choice__, config, random_state)

    if balancing_strategy == "weighting":
        if name_clf in ['decision_tree', 'extra_trees', 'liblinear_svc',
                        'libsvm_svc', "random_forest"]:
            model_clf.set_params(class_weight='balanced')
        if name_pre in ['liblinear_svc_preprocessor', 'extra_trees_preproc_for_classification']:
            model_pre.set_params(class_weight='balanced')

    resc_name, res_method = evaluation_rescaling(rescaling__choice__, config)
    enc_name, enc_method = evaluate_encoding(categorical_encoding__choice__, config, "all", is_sparse)

    list_preprocessing = []
    if len(categorical_features) > 0:
        list_preprocessing.append((enc_name, enc_method, categorical_features))
    if len(numerical_features) > 0:
        list_preprocessing.append((resc_name, res_method, numerical_features))

    preprocessing_pipeline = ColumnTransformer(transformers=list_preprocessing, remainder = "drop")

    pipeline_list = [
        evaluate_imputation(imputation_strategy),
        ("cleaning", preprocessing_pipeline),
        (name_pre, model_pre),
        (name_clf, model_clf)
    ]

    pipeline = Pipeline(pipeline_list)
    return pipeline, balancing_strategy == "weighting"


def evaluate(config, bestconfig, id_run, X=None, y=None, score_func=None, categorical_features=None, seed=None, test_data = {}, store_directory = ""):
    print("*", end="")
    try:
        from scipy.sparse import issparse
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import train_test_split
        import traceback
        import sys, os
        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X), seed)
            list_score_train = []
            list_score_test = []

            name_clf = pipeline.steps[3][0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.329, random_state = seed)

            fit_params = {}
            if balancing_strategy and name_clf in ['adaboost', 'gradient_boosting', 'random_forest', 'extra_trees',
                                                   'sgd', 'xgradient_boosting']:
                fit_params[name_clf + "__sample_weight"] = get_sample_weight(y_train)

            pipeline.fit(np.array(X_train), np.array(y_train), **fit_params)

            pred_valid = pipeline.predict(np.array(X_test))
            score = score_func(y_test, pipeline.predict(X_test))
            info = {"validation_score": score}

            if test_data:
                pred_test = pipeline.predict(np.array(test_data["X_test"]))
                info["test_score"] = score_func(test_data["y_test"], pred_test)
                #np.save(os.path.join(store_directory, "pred_test_{0}.npy".format(id_run)), pred_test)
                #np.save(os.path.join(store_directory, "pred_valid_{0}.npy".format(id_run)), pred_valid)

            return info

    except Exception as e: # TimeoutException
        print(sys.exc_info())
        raise(e)

    print(sys.exc_info())

    return {"validation_score": 0, "test_score": 0}


def evaluate_generate_metadata(config, bestconfig, id_run, X=None, y=None, score_func=None, categorical_features=None, seed=None):
    print("*", end="")
    try:
        from scipy.sparse import issparse
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import cross_val_score
        import traceback
        import sys, os
        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X), seed)
            list_score_train = []
            list_score_test = []

            name_clf = pipeline.steps[3][0]

            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.329, random_state = seed)

            fit_params = {}
            if balancing_strategy and name_clf in ['adaboost', 'gradient_boosting', 'random_forest', 'extra_trees',
                                                   'sgd', 'xgradient_boosting']:
                fit_params[name_clf + "__sample_weight"] = get_sample_weight(y)

            #pipeline.fit(np.array(X_train), np.array(y_train), **fit_params)
            #list_score.append(score_func(y_test, pipeline.predict(X_test)))
            scores = cross_val_score(pipeline, X, y, cv=StratifiedKFold(10), fit_params=fit_params, scoring="balanced_accuracy")
            #pred_valid = pipeline.predict(np.array(X_test))
            #score = score_func(y_test, pipeline.predict(X_test))
            info = {"validation_score": np.median(scores), "list_scores": list(scores)}

            return info

    except Exception as e: # TimeoutException
        print(sys.exc_info())
        raise(e)

    print(sys.exc_info())

    return {"validation_score": 0, "test_score": 0}



def test_function(config, X_train, y_train, X_test, y_test, categorical_features, random_state):
    from scipy.sparse import issparse
    from sklearn.metrics import balanced_accuracy_score
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    with warnings.catch_warnings():
        pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X_train), random_state)
        fit_params = {}
        name_clf = pipeline.steps[3][0]
        if balancing_strategy and name_clf in ['adaboost', 'gradient_boosting', 'random_forest', 'extra_trees', 'sgd',
                                               'xgradient_boosting']:
            fit_params[name_clf + "__sample_weight"] = get_sample_weight(y_train)
        pipeline.fit(X_train, y_train, **fit_params)
        y_pred = pipeline.predict(X_test)
        return balanced_accuracy_score(y_pred, y_test)


def run_pipeline(params):
    from sklearn.metrics import roc_auc_score
    from sklearn.base import clone
    model_, X, y, index, data_manager = params
    X_train, y_train = X[index[0]], y[index[0]]
    X_test, y_test = X[index[1]], y[index[1]]

    model = clone(model_)
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        score_ = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        score_ = roc_auc_score(y_test, model.predict(X_test))
    data_manager.add_data(score_, model)

    return (score_, model)


def evaluate_competition(config, bestconfig, X=None, y=None, score_func=None,
                         categorical_features=None, seed=None, data_manager=None,
                         time_limit_for_evaluation=None):
    print("-----------------------------------------------------------------------------------------------")
    print(config)
    try:
        from scipy.sparse import issparse
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from sklearn.model_selection import StratifiedKFold
        from sklearn.externals.joblib import Parallel, delayed
        from sklearn.base import clone
        from sklearn.utils import resample
        import random

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pipeline, balancing_strategy = config_to_pipeline(config, categorical_features, issparse(X))
            list_score = []

            name_clf = pipeline.steps[4][0]

            X_, y_ = resample(X, y, random_state=random.randint(0, 10000), n_samples=min([X.shape[0], 50000]))
            skf = StratifiedKFold(n_splits=3, random_state=seed).split(X_, y_)

            if bestconfig["score_validation"] == 0:
                r = Parallel(n_jobs=3, verbose=0, temp_folder="/tmp", backend="threading")(delayed(run_pipeline)((pipeline, X_, y_, index, data_manager)) for index in skf)
            else:
                r = Parallel(n_jobs=3, verbose=0, timeout=(time_limit_for_evaluation-2), temp_folder="/tmp", backend="threading")(delayed(run_pipeline)((pipeline, X_, y_, index, data_manager)) for index in skf)

            sum_score = 0
            for s, m in r:
                sum_score += s

            score = sum_score / 3
            print("Score", score)

            return {"validation_score": score}
    except TimeoutException as e:
        print("Timout!!!")
        raise(e)
    except MemorylimitException as e:
        print("Memout!!!")
        raise(e)
    except Exception as e:
        print(e)
        raise (e)
