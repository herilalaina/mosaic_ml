from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule

from sklearn import feature_selection, linear_model, ensemble, svm


def get_configuration_SelectPercentile():
    SelectPercentile = ListTask(is_ordered=False, name = "SelectPercentile",
                                  tasks = ["SelectPercentile__score_func",
                                           "SelectPercentile__percentile"])
    sampler = {
             "SelectPercentile__score_func": Parameter("SelectPercentile__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
             "SelectPercentile__percentile": Parameter("SelectPercentile__percentile", [1, 100], "uniform", "int")
    }
    rules = []
    return SelectPercentile, sampler, rules


def get_configuration_SelectKBest():
    SelectKBest = ListTask(is_ordered=False, name = "SelectKBest",
                                  tasks = ["SelectKBest__score_func",
                                           "SelectKBest__k"])
    sampler = {
             "SelectKBest__score_func": Parameter("SelectKBest__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
             "SelectKBest__k": Parameter("SelectKBest__k", [1, 20], "uniform", "int")
    }
    rules = []
    return SelectKBest, sampler, rules


def get_configuration_SelectFpr():
    SelectFpr = ListTask(is_ordered=False, name = "SelectFpr",
                                  tasks = ["SelectFpr__score_func",
                                           "SelectFpr__alpha"])
    sampler = {
             "SelectFpr__score_func": Parameter("SelectFpr__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
             "SelectFpr__alpha": Parameter("SelectFpr__alpha", [0, 0.05], "uniform", "float")
    }
    rules = []
    return SelectFpr, sampler, rules


def get_configuration_SelectFdr():
    SelectFdr = ListTask(is_ordered=False, name = "SelectFdr",
                                  tasks = ["SelectFdr__score_func",
                                           "SelectFdr__alpha"])
    sampler = {
             "SelectFdr__score_func": Parameter("SelectFdr__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
             "SelectFdr__alpha": Parameter("SelectFdr__alpha", [0, 0.05], "uniform", "float")
    }
    rules = []
    return SelectFdr, sampler, rules


def get_configuration_SelectFwe():
    SelectFwe = ListTask(is_ordered=False, name = "SelectFwe",
                                  tasks = ["SelectFwe__score_func",
                                           "SelectFwe__alpha"])
    sampler = {
             "SelectFwe__score_func": Parameter("SelectFwe__score_func", [feature_selection.f_classif, feature_selection.mutual_info_classif, feature_selection.chi2], "choice", "func"),
             "SelectFwe__alpha": Parameter("SelectFwe__alpha", [0, 0.05], "uniform", "float")
    }
    rules = []
    return SelectFwe, sampler, rules


def get_configuration_SelectFromModel():
    SelectFromModel = ListTask(is_ordered=False, name = "SelectFromModel",
                                  tasks = ["SelectFromModel__estimator",
                                           "SelectFromModel__threshold"])
    sampler = {
             "SelectFromModel__estimator": Parameter("SelectFromModel__estimator", [svm.LinearSVC(penalty='l1', dual=False), ensemble.ExtraTreesClassifier(n_estimators=100, criterion="gini"), ensemble.RandomForestClassifier(n_estimators=100)], "choice", "func"),
             "SelectFromModel__threshold": Parameter("SelectFromModel__threshold", ["mean", "median"], "choice", "string")
    }
    rules = []
    return SelectFromModel, sampler, rules

def get_configuration_LinearSVCPrep():
    rules = [
        #ValueRule([("LinearSVCPrep__loss", "hinge"), ("LinearSVCPrep__penalty", "l2"), ("LinearSVCPrep__dual", True)]),
        #ValueRule([("LinearSVCPrep__loss", "hinge"), ("LinearSVCPrep__penalty", "l2")])
    ]
    LinearSVCPrep = ListTask(is_ordered=False, name = "LinearSVCPrep",
                                  tasks = ["LinearSVCPrep__penalty",
                                           "LinearSVCPrep__loss",
                                           "LinearSVCPrep__dual",
                                           "LinearSVCPrep__tol",
                                           "LinearSVCPrep__C",
                                           "LinearSVCPrep__class_weight",
                                           "LinearSVCPrep__max_iter"],
                                  rules = rules)
    sampler = {
           "LinearSVCPrep__penalty": Parameter("LinearSVCPrep__penalty", "l1", "constant", "string"),
           "LinearSVCPrep__loss": Parameter("LinearSVCPrep__loss", "squared_hinge", "constant", "string"),
           "LinearSVCPrep__dual": Parameter("LinearSVCPrep__dual", False, "constant", "bool"),
           "LinearSVCPrep__tol": Parameter("LinearSVCPrep__tol", [1e-5, 1e-1], "log_uniform", "bool"),
           "LinearSVCPrep__C": Parameter("LinearSVCPrep__C", [0.03125, 30], "log_uniform", "float"),
           "LinearSVCPrep__class_weight": Parameter("LinearSVCPrep__class_weight", "balanced", "constant", "string"),
           "LinearSVCPrep__max_iter": Parameter("LinearSVCPrep__max_iter", [1, 100], "uniform", "int")
    }
    return LinearSVCPrep, sampler, rules

def get_configuration_ExtraTreesClassifierPrep():
    ExtraTreesClassifierPrep = ListTask(is_ordered=False, name = "ExtraTreesClassifierPrep",
                                  tasks = ["ExtraTreesClassifierPrep__n_estimators",
                                           "ExtraTreesClassifierPrep__criterion",
                                           "ExtraTreesClassifierPrep__max_depth",
                                           "ExtraTreesClassifierPrep__min_samples_split",
                                           "ExtraTreesClassifierPrep__min_samples_leaf",
                                           "ExtraTreesClassifierPrep__min_weight_fraction_leaf",
                                           "ExtraTreesClassifierPrep__max_features",
                                           "ExtraTreesClassifierPrep__max_leaf_nodes",
                                           #"ExtraTreesClassifierPrep__min_impurity_decrease",
                                           "ExtraTreesClassifierPrep__class_weight",
                                           #"ExtraTreesClassifierPrep__bootstrap",
                                           "ExtraTreesClassifierPrep__n_jobs",
                                           #"ExtraTreesClassifierPrep__oob_score",
                                           #"ExtraTreesClassifierPrep__warm_start"
                                           ])
    sampler = {
             "ExtraTreesClassifierPrep__n_estimators": Parameter("ExtraTreesClassifierPrep__n_estimators", [50, 700], "uniform", "int"),
             "ExtraTreesClassifierPrep__criterion": Parameter("ExtraTreesClassifierPrep__criterion", ["gini", "entropy"], "choice", "string"),
             "ExtraTreesClassifierPrep__max_depth": Parameter("ExtraTreesClassifierPrep__max_depth", [1, 10], "uniform", "int"),
             "ExtraTreesClassifierPrep__min_samples_split": Parameter("ExtraTreesClassifierPrep__min_samples_split", [2, 20], "uniform", "int"),
             "ExtraTreesClassifierPrep__min_samples_leaf": Parameter("ExtraTreesClassifierPrep__min_samples_leaf", [1, 20], "uniform", "int"),
             "ExtraTreesClassifierPrep__min_weight_fraction_leaf": Parameter("ExtraTreesClassifierPrep__min_weight_fraction_leaf", 0.0, "constant", "float"),
             "ExtraTreesClassifierPrep__max_features": Parameter("ExtraTreesClassifierPrep__max_features", ["auto", "sqrt", "log2", None], "choice", "string"),
             "ExtraTreesClassifierPrep__max_leaf_nodes": Parameter("ExtraTreesClassifierPrep__max_leaf_nodes", None, "constant", "string"),
             #"ExtraTreesClassifierPrep__min_impurity_decrease": Parameter("ExtraTreesClassifierPrep__min_impurity_decrease", [0, 0.05], "uniform", "float"),
             "ExtraTreesClassifierPrep__class_weight": Parameter("ExtraTreesClassifierPrep__class_weight", "balanced", "constant", "string"),
             #"ExtraTreesClassifierPrep__bootstrap": Parameter("ExtraTreesClassifierPrep__bootstrap", [True, False], "choice", "bool"),
             "ExtraTreesClassifierPrep__n_jobs": Parameter("ExtraTreesClassifierPrep__n_jobs", -1, "constant", "int"),
             #"ExtraTreesClassifierPrep__oob_score": Parameter("ExtraTreesClassifierPrep__oob_score", [True, False], "choice", "bool"),
             #"ExtraTreesClassifierPrep__warm_start": Parameter("ExtraTreesClassifierPrep__warm_start", [True, False], "choice", "bool")
    }

    rules = [
        # ValueRule([("ExtraTreesClassifierPrep__oob_score", True), ("ExtraTreesClassifierPrep__bootstrap", True), ("ExtraTreesClassifierPrep__warm_start", False)])
    ]
    return ExtraTreesClassifierPrep, sampler, rules


def get_configuration_RFE():
    RFE = ListTask(is_ordered=False, name = "RFE",
                                  tasks = ["RFE__estimator",
                                           "RFE__n_features_to_select",
                                           "RFE__step"])
    sampler = {
             "RFE__estimator": Parameter("RFE__estimator", [linear_model.RidgeClassifier(), svm.LinearSVC(), ensemble.ExtraTreesClassifier(), ensemble.RandomForestClassifier()], "choice", "func"),
             "RFE__n_features_to_select": Parameter("RFE__n_features_to_select", [1, 20], "uniform", "int"),
             "RFE__step": Parameter("RFE__step", [1, 20], "uniform", "int"),
    }
    rules = []
    return RFE, sampler, rules
