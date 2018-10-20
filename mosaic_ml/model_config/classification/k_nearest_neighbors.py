from sklearn.neighbors import KNeighborsClassifier


def get_model(name, config):
    model = KNeighborsClassifier(
        n_neighbors=int(config["classifier:k_nearest_neighbors:n_neighbors"]),
        p=int(config["classifier:k_nearest_neighbors:p"]),
        weights=config["classifier:k_nearest_neighbors:weights"]
    )
    return (name, model)
