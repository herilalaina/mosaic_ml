from sklearn.ensemble import RandomTreesEmbedding


def get_model(name, config):
    model = RandomTreesEmbedding(
        max_depth=int(config["preprocessor:random_trees_embedding:max_depth"]),
        max_leaf_nodes=None,
        min_samples_leaf=int(config["preprocessor:random_trees_embedding:min_samples_leaf"]),
        min_samples_split=int(config["preprocessor:random_trees_embedding:min_samples_split"]),
        min_weight_fraction_leaf=float(config["preprocessor:random_trees_embedding:min_weight_fraction_leaf"]),
        n_estimators=int(config["preprocessor:random_trees_embedding:n_estimators"])
    )
    return (name, model)