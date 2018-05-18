from mosaic_ml.model.data_preprocessing.decomposition import *
from mosaic_ml.model.data_preprocessing.feature_selection import *
from mosaic_ml.model.data_preprocessing.cluster import *
from mosaic_ml.model.classification.discriminant_analysis import *
from mosaic_ml.model.classification.dummy import *
from mosaic_ml.model.classification.ensemble import *
from mosaic_ml.model.classification.gaussian_process import *
from mosaic_ml.model.classification.linear_model import *
from mosaic_ml.model.classification.naive_bayes import *
from mosaic_ml.model.classification.neighbors import *
from mosaic_ml.model.classification.neural_network import *
from mosaic_ml.model.classification.semi_supervised import *
from mosaic_ml.model.classification.svm import *
from mosaic_ml.model.classification.tree import *

from sklearn import discriminant_analysis, dummy, ensemble, gaussian_process, linear_model, naive_bayes, cluster
from sklearn import neighbors, neural_network, semi_supervised, svm, tree, decomposition, feature_selection

Ressource_parameters = [
    "AdaBoostClassifier__n_estimators",
    "BaggingClassifier__n_estimators",
    "ExtraTreesClassifier__n_estimators",
    "GradientBoostingClassifier__n_estimators",
    "RandomForestClassifier__n_estimators",
    "SGDClassifier__max_iter",
    "RidgeClassifier__max_iter",
    "Perceptron__max_iter",
    "PassiveAggressiveClassifier__max_iter",
    "MLPClassifier__max_iter",
    "LinearSVC__max_iter",
    "NuSVC__max_iter",
    "SVC__max_iter"
]



DATA_DEPENDANT_PARAMS = [
    # Classifier
    "ExtraTreesClassifier__max_depth",
    "GradientBoostingClassifier__max_depth",
    "RandomForestClassifier__max_depth",
    "KNeighborsClassifier__n_neighbors",
    #"LabelPropagation__n_neighbors",
    #"LabelSpreading__n_neighbors",
    "DecisionTreeClassifier__max_depth",
    "ExtraTreeClassifier__max_depth",

    # Data preprocessing
    "DictionaryLearning__n_components",
    "FactorAnalysis__n_components",
    "FastICA__n_components",
    "IncrementalPCA__n_components",
    "KernelPCA__n_components",
    "LatentDirichletAllocation__n_components",
    "MiniBatchDictionaryLearning__n_components",
    "MiniBatchSparsePCA__n_components",
    "NMF__n_components",
    "PCA__n_components",
    "SelectKBest__k",
    "RFE__n_features_to_select"
]


list_n_jobs_parms = [
    #"BaggingClassifier__n_jobs",
    "ExtraTreesClassifier__n_jobs",
    "RandomForestClassifier__n_jobs",
    # "GaussianProcessClassifier__n_jobs",
    "Perceptron__n_jobs",
    "PassiveAggressiveClassifier__n_jobs",
    "KNeighborsClassifier__n_jobs",
    "DictionaryLearning__n_jobs",
    "KernelPCA__n_jobs",
    "LatentDirichletAllocation__n_jobs",
    "MiniBatchDictionaryLearning__n_jobs"
]

list_available_classifiers = {
    "LinearDiscriminantAnalysis": discriminant_analysis.LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis,
    # Dummy
    "DummyClassifier": dummy.DummyClassifier,
    # Ensemble
    "AdaBoostClassifier": ensemble.AdaBoostClassifier,
    #"BaggingClassifier": ensemble.BaggingClassifier,
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
    # gaussian_process
    #"GaussianProcessClassifier": gaussian_process.GaussianProcessClassifier,
    # linear_model
    # "LogisticRegression": linear_model.LogisticRegression,
    "Perceptron": linear_model.Perceptron,
    "SGDClassifier": linear_model.SGDClassifier,
    "RidgeClassifier": linear_model.RidgeClassifier,
    "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier,
    # naive_bayes
    "GaussianNB": naive_bayes.GaussianNB,
    "MultinomialNB": naive_bayes.MultinomialNB,
    # neighbors
    "KNeighborsClassifier": neighbors.KNeighborsClassifier,
    # "RadiusNeighborsClassifier": neighbors.RadiusNeighborsClassifier,
    # neural_network
    "MLPClassifier": neural_network.MLPClassifier,
    # semi_supervised
    #"LabelSpreading": semi_supervised.LabelSpreading,
    #"LabelPropagation": semi_supervised.LabelPropagation,
    # svm
    "SVC": svm.SVC,
    # "NuSVC": svm.NuSVC,
    "LinearSVC": svm.LinearSVC,
    # tree
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    "ExtraTreeClassifier": tree.ExtraTreeClassifier,
}


list_available_preprocessing = {
    #"DictionaryLearning": decomposition.DictionaryLearning,
    "PCA": decomposition.PCA,
    "NMF": decomposition.NMF,
    "FastICA": decomposition.FastICA,
    "KernelPCA": decomposition.KernelPCA,
    "FactorAnalysis": decomposition.FactorAnalysis,
    "IncrementalPCA": decomposition.IncrementalPCA,
    #"MiniBatchSparsePCA": decomposition.MiniBatchSparsePCA,
    #"MiniBatchDictionaryLearning": decomposition.MiniBatchDictionaryLearning,
    "LatentDirichletAllocation": decomposition.LatentDirichletAllocation,
    # feature_selection
    "RFE": feature_selection.RFE,
    "SelectFdr": feature_selection.SelectFdr,
    "SelectFpr": feature_selection.SelectFpr,
    "SelectFwe": feature_selection.SelectFwe,
    "SelectKBest": feature_selection.SelectKBest,
    "SelectPercentile": feature_selection.SelectPercentile,
    "SelectFromModel": feature_selection.SelectFromModel,
    # Cluster
    "FeatureAgglomeration": cluster.FeatureAgglomeration
}


def get_all_classifier():
    return [
        # Discrimiant analysis
        get_configuration_LinearDiscriminantAnalysis,
        get_configuration_QuadraticDiscriminantAnalysis,

        # Dummy
        get_configuration_DummyClassifier,

        # Ensemble
        get_configuration_AdaBoostClassifier,
        #get_configuration_BaggingClassifier,
        get_configuration_ExtraTreesClassifier,
        get_configuration_RandomForestClassifier,
        get_configuration_GradientBoostingClassifier,

        # gaussian_process
        # get_configuration_GaussianProcessClassifier,

        # linear_model
        # get_configuration_LogisticRegression,
        get_configuration_Perceptron,
        get_configuration_SGDClassifier,
        get_configuration_RidgeClassifier,
        get_configuration_PassiveAggressiveClassifier,

        # naive_bayes
        get_configuration_GaussianNB,
        get_configuration_MultinomialNB,

        # neighbors
        get_configuration_KNeighborsClassifier,
        # get_configuration_RadiusNeighborsClassifier,

        # neural_network
        get_configuration_MLPClassifier,

        # semi_supervised
        #get_configuration_LabelSpreading,
        #get_configuration_LabelPropagation,

        # svm
        get_configuration_SVC,
        # get_configuration_NuSVC,
        get_configuration_LinearSVC,

        # tree
        get_configuration_DecisionTreeClassifier,
        get_configuration_ExtraTreeClassifier
    ]


def get_all_data_preprocessing():
    return [
        # decomposition
        #get_configuration_DictionaryLearning,
        get_configuration_PCA,
        get_configuration_NMF,
        get_configuration_FastICA,
        get_configuration_KernelPCA,
        get_configuration_FactorAnalysis,
        get_configuration_IncrementalPCA,
        #get_configuration_MiniBatchSparsePCA,
        #get_configuration_MiniBatchDictionaryLearning,
        get_configuration_LatentDirichletAllocation,

        # feature_selection
        get_configuration_RFE,
        get_configuration_SelectFdr,
        get_configuration_SelectFpr,
        get_configuration_SelectFwe,
        get_configuration_SelectKBest,
        get_configuration_SelectPercentile,
        get_configuration_SelectFromModel,

        # Cluster
        get_configuration_FeatureAgglomeration
    ]
