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
