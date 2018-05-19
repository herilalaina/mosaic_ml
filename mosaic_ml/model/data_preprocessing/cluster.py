from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_FeatureAgglomeration():
    rules = [
        ValueRule([("FeatureAgglomeration__linkage", "ward"), ("FeatureAgglomeration__affinity", "euclidean")])
    ]
    FeatureAgglomeration = ListTask(is_ordered=False, name = "FeatureAgglomeration",
                                  tasks = ["FeatureAgglomeration__n_clusters",
                                           "FeatureAgglomeration__affinity",
                                           "FeatureAgglomeration__linkage"],
                                  rules = rules)
    sampler = {
         "FeatureAgglomeration__n_clusters": Parameter("FeatureAgglomeration__n_clusters", [2, 30], 'uniform', "int"),
         "FeatureAgglomeration__affinity": Parameter("FeatureAgglomeration__affinity", ["l1", "l2", "manhattan", "cosine", "euclidean"], "choice", "string"),
         "FeatureAgglomeration__linkage": Parameter("FeatureAgglomeration__linkage", ["ward", "complete", "average"], "choice", "string")
    }

    return FeatureAgglomeration, sampler, rules
