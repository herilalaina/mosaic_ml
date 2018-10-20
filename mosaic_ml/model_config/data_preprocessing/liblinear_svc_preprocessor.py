from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC


def get_model(name, config):
    model = SelectFromModel(
        estimator=LinearSVC(
            C=float(config["preprocessor:liblinear_svc_preprocessor:C"]),
            dual=eval(config["preprocessor:liblinear_svc_preprocessor:dual"]),
            fit_intercept=eval(config["preprocessor:liblinear_svc_preprocessor:fit_intercept"]),
            intercept_scaling=int(config["preprocessor:liblinear_svc_preprocessor:intercept_scaling"]),
            loss=config["preprocessor:liblinear_svc_preprocessor:loss"],
            multi_class=config["preprocessor:liblinear_svc_preprocessor:multi_class"],
            penalty=config["preprocessor:liblinear_svc_preprocessor:penalty"],
            tol=float(config["preprocessor:liblinear_svc_preprocessor:tol"])
        )
    )
    return (name, model)
