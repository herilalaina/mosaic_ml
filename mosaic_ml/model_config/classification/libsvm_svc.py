from sklearn.svm import SVC

def get_model(name, config):
    model = SVC(
        C=float(config["classifier:libsvm_svc:C"]),
        gamma=float(config["classifier:libsvm_svc:gamma"]),
        kernel=config["classifier:libsvm_svc:kernel"],
        max_iter=int(config["classifier:libsvm_svc:max_iter"]),
        shrinking=eval(config["classifier:libsvm_svc:shrinking"]),
        tol=float(config["classifier:libsvm_svc:tol"])
    )
    return (name, model)