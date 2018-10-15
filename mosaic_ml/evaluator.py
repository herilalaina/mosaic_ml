from mosaic_ml.model_config.classification import get_classifier
from mosaic_ml.model_config.data_preprocessing import get_data_preprocessing

def evaluate_imputation(imputation_strategy):
    from sklearn.preprocessing import Imputer

    imp = Imputer(strategy=imputation_strategy)
    return ("Imputation", imp)


def evaluate_encoding(choice, config):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import FunctionTransformer

    if choice == "no_encoding":
        encoding = FunctionTransformer()
    elif choice == "one_hot_encoding":
        encoding = OneHotEncoder(sparse=False)
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

def evaluate(config, bestconfig, X=None, y=None, X_TEST=None, Y_TEST=None, info = {}, score_func=None):
    print("*", end="")
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with warnings.catch_warnings():
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import StratifiedKFold
            import numpy as np

            warnings.simplefilter("ignore")

            list_params = config.keys()

            balancing_strategy = config["balancing:strategy"]
            imputation_strategy = config["imputation:strategy"]
            categorical_encoding__choice__ = config["categorical_encoding:__choice__"]
            rescaling__choice__ = config["rescaling:__choice__"]

            classifier__choice__ = config["classifier:__choice__"]
            preprocessor__choice__ = config["preprocessor:__choice__"]

            pipeline_list = [
                evaluate_imputation(imputation_strategy),
                evaluate_encoding(categorical_encoding__choice__, config),
                evaluation_rescaling(rescaling__choice__, config),
                get_data_preprocessing.evaluate(preprocessor__choice__, config),
                get_classifier.evaluate_classifier(classifier__choice__, config)
            ]

            list_score = []
            pipeline = Pipeline(pipeline_list)

            pred_list = []

            skf = StratifiedKFold(n_splits=2, random_state=42)
            for train_index, test_index in skf.split(X, y):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
                pipeline.fit(X_train, y_train)
                list_score.append(score_func(y_test, pipeline.predict(X_test)))

                try:
                    pred_list.append(pipeline.predict_proba(X_TEST))
                except Exception as e:
                    pred_list.append(pipeline.predict(X_TEST))

                if list_score[-1] < bestconfig["score_validation"]:
                    return {"validation_score": list_score[-1], "test_score": 0}

            test_preds = np.mean(pred_list, axis=0)
            if len(np.shape(test_preds)) == 2:
                test_score_ = np.mean(pred_list, axis=0)
                test_score = np.argmax(test_score_, 1)
            else:
                test_score = np.round(test_preds)

            score_test = score_func(Y_TEST, test_score)

            return {"validation_score": min(list_score), "test_score": score_test}
    except Exception as e:
        return {"validation_score": 0, "test_score": 0}



