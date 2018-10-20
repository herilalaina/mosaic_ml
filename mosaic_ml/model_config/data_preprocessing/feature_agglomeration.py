from sklearn.cluster import FeatureAgglomeration


def get_model(name, config):
    model = FeatureAgglomeration(
        affinity=config["preprocessor:feature_agglomeration:affinity"],
        linkage=config["preprocessor:feature_agglomeration:linkage"],
        n_clusters=int(config["preprocessor:feature_agglomeration:n_clusters"]),
        pooling_func=config["preprocessor:feature_agglomeration:pooling_func"]
    )
    return (name, model)
