from sklearn.svm import LinearSVC

def get_model(choice, config):
    model = LinearSVC(
        C=float(config["classifier:liblinear_svc:C"]),
        dual=eval(config["classifier:liblinear_svc:dual"]),
        fit_intercept=eval(config["classifier:liblinear_svc:fit_intercept"]),
        intercept_scaling=int(config["classifier:liblinear_svc:intercept_scaling"]),
        loss=config["classifier:liblinear_svc:loss"],
        multi_class=config["classifier:liblinear_svc:multi_class"],
        penalty=config["classifier:liblinear_svc:penalty"],
        tol=float(config["classifier:liblinear_svc:tol"])
    )
    return (choice, model)