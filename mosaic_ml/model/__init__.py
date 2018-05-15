from mosaic_ml.model.classification import discriminant_analysis, dummy, ensemble
from mosaic_ml.model.classification import gaussian_process, linear_model, naive_bayes
from mosaic_ml.model.classification import neighbors, neural_network, semi_supervised
from mosaic_ml.model.classification import svm, tree
from mosaic_ml.model.data_preprocessing import decomposition, feature_selection

DATA_DEPENDANT_PARAMS = [
    # Classifier
    "ExtraTreesClassifier__max_depth",
    "GradientBoostingClassifier__max_depth",
    "RandomForestClassifier__max_depth",
    "KNeighborsClassifier__n_neighbors",
    "LabelPropagation__n_neighbors",
    "LabelSpreading__n_neighbors",
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


def get_all_classifier():
    return [
        # Discrimiant analysis
        discriminant_analysis.get_configuration_LinearDiscriminantAnalysis,
        discriminant_analysis.get_configuration_QuadraticDiscriminantAnalysis,

        # Dummy
        dummy.get_configuration_DummyClassifier,

        # Ensemble
        ensemble.get_configuration_AdaBoostClassifier,
        ensemble.get_configuration_BaggingClassifier,
        ensemble.get_configuration_ExtraTreesClassifier,
        ensemble.get_configuration_RandomForestClassifier,
        ensemble.get_configuration_GradientBoostingClassifier,

        # gaussian_process
        gaussian_process.get_configuration_GaussianProcessClassifier,

        # linear_model
        linear_model.get_configuration_LogisticRegression,
        linear_model.get_configuration_Perceptron,
        linear_model.get_configuration_SGDClassifier,
        linear_model.get_configuration_RidgeClassifier,
        linear_model.get_configuration_PassiveAggressiveClassifier,

        # naive_bayes
        naive_bayes.get_configuration_GaussianNB,
        naive_bayes.get_configuration_MultinomialNB,

        # neighbors
        neighbors.get_configuration_KNeighborsClassifier,
        neighbors.get_configuration_RadiusNeighborsClassifier,

        # neural_network
        neural_network.get_configuration_MLPClassifier,

        # semi_supervised
        semi_supervised.get_configuration_LabelSpreading,
        semi_supervised.get_configuration_LabelPropagation,

        # svm
        svm.get_configuration_SVC,
        svm.get_configuration_NuSVC,
        svm.get_configuration_LinearSVC,

        # tree
        tree.get_configuration_DecisionTreeClassifier,
        tree.get_configuration_ExtraTreeClassifier
    ]


def get_all_data_preprocessing():
    return [
        # decomposition
        decomposition.get_configuration_DictionaryLearning,
        decomposition.get_configuration_PCA,
        decomposition.get_configuration_NMF,
        decomposition.get_configuration_FastICA,
        decomposition.get_configuration_KernelPCA,
        decomposition.get_configuration_FactorAnalysis,
        decomposition.get_configuration_IncrementalPCA,
        decomposition.get_configuration_MiniBatchSparsePCA,
        decomposition.get_configuration_MiniBatchDictionaryLearning,
        decomposition.get_configuration_LatentDirichletAllocation,

        # feature_selection
        feature_selection.get_configuration_RFE,
        feature_selection.get_configuration_SelectFdr,
        feature_selection.get_configuration_SelectFpr,
        feature_selection.get_configuration_SelectFwe,
        feature_selection.get_configuration_SelectKBest,
        feature_selection.get_configuration_SelectPercentile,
        feature_selection.get_configuration_SelectFromModel
    ]
