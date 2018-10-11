from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_model(name, config):
    model = LinearDiscriminantAnalysis(
        n_components=int(config["classifier:lda:n_components"]),
        #shrinkage= config["classifier:lda:shrinkage"] if config["classifier:lda:shrinkage"] != "None" else None,
        tol=float(config["classifier:lda:tol"])
    )
    return (name, model)