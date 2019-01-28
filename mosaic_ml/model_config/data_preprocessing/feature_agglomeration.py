from sklearn.cluster import FeatureAgglomeration
import numpy as np

def get_model(name, config):
    if config["preprocessor:feature_agglomeration:pooling_func"] == "mean":
        pooling_func_ = np.mean
    elif config["preprocessor:feature_agglomeration:pooling_func"] == "max":
        pooling_func_ = np.max
    else:
        pooling_func_ = np.median

    model = FeatureAgglomeration(
        affinity=config["preprocessor:feature_agglomeration:affinity"],
        linkage=config["preprocessor:feature_agglomeration:linkage"],
        n_clusters=int(config["preprocessor:feature_agglomeration:n_clusters"]),
        pooling_func=pooling_func_
    )
    return (name, model)
