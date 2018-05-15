from mosaic.space import Parameter
from mosaic.scenario import ListTask
from mosaic.rules import ChildRule, ValueRule


def get_configuration_DictionaryLearning():
    DictionaryLearning = ListTask(is_ordered=False, name = "DictionaryLearning",
                                  tasks = ["DictionaryLearning__n_components",
                                           "DictionaryLearning__alpha",
                                           "DictionaryLearning__max_iter",
                                           "DictionaryLearning__tol",
                                           "DictionaryLearning__fit_algorithm",
                                           "DictionaryLearning__transform_algorithm",
                                           "DictionaryLearning__transform_alpha",
                                           "DictionaryLearning__n_jobs",
                                           "DictionaryLearning__split_sign"])
    sampler = {
         "DictionaryLearning__n_components": Parameter("DictionaryLearning__n_components", [2, 20], 'uniform', "int"),
         "DictionaryLearning__alpha": Parameter("DictionaryLearning__alpha", [0, 1], "uniform", "float"),
         "DictionaryLearning__max_iter": Parameter("DictionaryLearning__max_iter", [1, 100], "uniform", "int"),
         "DictionaryLearning__tol": Parameter("DictionaryLearning__tol", [0, 0.1], "uniform", "float"),
         "DictionaryLearning__fit_algorithm": Parameter("DictionaryLearning__fit_algorithm", ["lars", "cd"], "choice", "string"),
         "DictionaryLearning__transform_algorithm": Parameter("DictionaryLearning__transform_algorithm", ["lasso_lars", "lasso_cd", "lars", "omp", "threshold"], "choice", "string"),
         "DictionaryLearning__transform_alpha": Parameter("DictionaryLearning__transform_alpha", [0, 1], "uniform", "float"),
         "DictionaryLearning__n_jobs": Parameter("DictionaryLearning__n_jobs", -1, "constant", "int"),
         "DictionaryLearning__split_sign": Parameter("DictionaryLearning__split_sign", [True, False], "choice", "bool")
    }

    rules = []
    return DictionaryLearning, sampler, rules


def get_configuration_FactorAnalysis():
    FactorAnalysis = ListTask(is_ordered=False, name = "FactorAnalysis",
                                  tasks = ["FactorAnalysis__n_components",
                                           "FactorAnalysis__noise_variance_init",
                                           "FactorAnalysis__max_iter",
                                           "FactorAnalysis__tol",
                                           "FactorAnalysis__svd_method",
                                           "FactorAnalysis__iterated_power"])
    sampler = {
          "FactorAnalysis__n_components": Parameter("FactorAnalysis__n_components", [2, 20], "uniform", 'int'),
          "FactorAnalysis__noise_variance_init": Parameter("FactorAnalysis__noise_variance_init", None, "constant", "string"),
          "FactorAnalysis__max_iter": Parameter("FactorAnalysis__max_iter", [1, 100], "uniform", "int"),
          "FactorAnalysis__tol": Parameter("FactorAnalysis__tol", [0, 0.1], "uniform", "float"),
          "FactorAnalysis__svd_method": Parameter("FactorAnalysis__svd_method", ["lapack", "randomized"], "choice", "string"),
          "FactorAnalysis__iterated_power": Parameter("FactorAnalysis__iterated_power", [1, 5], "uniform", "int")
    }

    rules = [
        ChildRule(applied_to = ["FactorAnalysis__iterated_power"], parent = "FactorAnalysis__svd_method", value = ["randomized"])
    ]
    return FactorAnalysis, sampler, rules


def get_configuration_FastICA():
    FastICA = ListTask(is_ordered=False, name = "FastICA",
                                  tasks = ["FastICA__n_components",
                                           "FastICA__algorithm",
                                           "FastICA__max_iter",
                                           "FastICA__tol",
                                           "FastICA__whiten",
                                           "FastICA__fun"])
    sampler = {
          "FastICA__n_components": Parameter("FastICA__n_components", [2, 20], "uniform", 'int'),
          "FastICA__algorithm": Parameter("FastICA__algorithm", ["parallel", "deflation"], "choice", "string"),
          "FastICA__max_iter": Parameter("FastICA__max_iter", [1, 100], "uniform", "int"),
          "FastICA__tol": Parameter("FastICA__tol", [0, 0.1], "uniform", "float"),
          "FastICA__whiten": Parameter("FastICA__whiten", [True, False], "choice", "bool"),
          "FastICA__fun": Parameter("FastICA__fun", ["logcosh", "exp", "cube"], "choice", "int")
    }

    rules = []
    return FastICA, sampler, rules


def get_configuration_IncrementalPCA():
    IncrementalPCA = ListTask(is_ordered=False, name = "IncrementalPCA",
                                  tasks = ["IncrementalPCA__n_components",
                                           "IncrementalPCA__whiten",
                                           "IncrementalPCA__batch_size"])
    sampler = {
          "IncrementalPCA__n_components": Parameter("IncrementalPCA__n_components", [2, 20], "uniform", 'int'),
          "IncrementalPCA__whiten": Parameter("IncrementalPCA__whiten", [True, False], "choice", "bool"),
          "IncrementalPCA__batch_size": Parameter("IncrementalPCA__batch_size", [10, 400], "uniform", "int")
    }

    rules = []
    return IncrementalPCA, sampler, rules


def get_configuration_KernelPCA():
    KernelPCA = ListTask(is_ordered=False, name = "KernelPCA",
                                  tasks = ["KernelPCA__n_components",
                                           "KernelPCA__kernel",
                                           "KernelPCA__gamma",
                                           "KernelPCA__degree",
                                           "KernelPCA__coef0",
                                           "KernelPCA__alpha",
                                           "KernelPCA__eigen_solver",
                                           "KernelPCA__tol",
                                           "KernelPCA__max_iter",
                                           "KernelPCA__remove_zero_eig",
                                           "KernelPCA__n_jobs"])
    sampler = {
          "KernelPCA__n_components": Parameter("KernelPCA__n_components", [2, 20], "uniform", 'int'),
          "KernelPCA__kernel": Parameter("KernelPCA__kernel", ["linear", "poly", "rbf", "sigmoid", "cosine"], "choice", "string"),
          "KernelPCA__batch_size": Parameter("KernelPCA__batch_size", [0, 1], "uniform", "float"),
          "KernelPCA__degree": Parameter("KernelPCA__degree", [2, 3, 4, 5], "choice", "int"),
          "KernelPCA__coef0": Parameter("KernelPCA__coef0", [0, 1], "uniform", "float"),
          "KernelPCA__alpha": Parameter("KernelPCA__alpha", [0, 1], "uniform", "float"),
          "KernelPCA__eigen_solver": Parameter("KernelPCA__eigen_solver", ["auto", "dense", "arpack"], "choice", "string"),
          "KernelPCA__tol": Parameter("KernelPCA__tol", [0, 1], "uniform", "float"),
          "KernelPCA__max_iter": Parameter("KernelPCA__max_iter", [1, 100], "uniform", "int"),
          "KernelPCA__remove_zero_eig": Parameter("KernelPCA__remove_zero_eig", [True, False], "uniform", "bool"),
          "KernelPCA__n_jobs": Parameter("KernelPCA__n_jobs", -1, "constant", "int"),
    }

    rules = [
        ChildRule(applied_to = ["KernelPCA__degree"], parent = "KernelPCA__kernel", value = ["poly"]),
        ChildRule(applied_to = ["KernelPCA__gamma"], parent = "KernelPCA__kernel", value = ["poly", "rbf", "sigmoid"]),
        ChildRule(applied_to = ["KernelPCA__coef0"], parent = "KernelPCA__kernel", value = ["poly", "sigmoid"]),
    ]
    return KernelPCA, sampler, rules



def get_configuration_LatentDirichletAllocation():
    LatentDirichletAllocation = ListTask(is_ordered=False, name = "LatentDirichletAllocation",
                                  tasks = ["LatentDirichletAllocation__n_components",
                                           "LatentDirichletAllocation__doc_topic_prior",
                                           "LatentDirichletAllocation__topic_word_prior",
                                           "LatentDirichletAllocation__learning_method",
                                           "LatentDirichletAllocation__learning_decay",
                                           "LatentDirichletAllocation__learning_offset",
                                           "LatentDirichletAllocation__max_iter",
                                           "LatentDirichletAllocation__batch_size",
                                           "LatentDirichletAllocation__evaluate_every",
                                           "LatentDirichletAllocation__mean_change_tol",
                                           "LatentDirichletAllocation__n_jobs"])
    sampler = {
          "LatentDirichletAllocation__n_components": Parameter("LatentDirichletAllocation__n_components", [2, 20], "uniform", 'int'),
          "LatentDirichletAllocation__doc_topic_prior": Parameter("LatentDirichletAllocation__doc_topic_prior", [0, 1], "uniform", "float"),
          "LatentDirichletAllocation__topic_word_prior": Parameter("LatentDirichletAllocation__topic_word_prior", [0, 1], "uniform", "float"),
          "LatentDirichletAllocation__learning_method": Parameter("LatentDirichletAllocation__learning_method", ["batch", "online"], "choice", "string"),
          "LatentDirichletAllocation__learning_decay": Parameter("LatentDirichletAllocation__learning_decay", [0.51, 1], "uniform", "float"),
          "LatentDirichletAllocation__learning_offset": Parameter("LatentDirichletAllocation__learning_offset", [1, 20], "uniform", "float"),
          "LatentDirichletAllocation__max_iter": Parameter("LatentDirichletAllocation__max_iter", [1, 100], "uniform", "int"),
          "LatentDirichletAllocation__batch_size": Parameter("LatentDirichletAllocation__batch_size", [75, 400], "uniform", "int"),
          "LatentDirichletAllocation__evaluate_every": Parameter("LatentDirichletAllocation__evaluate_every", 0, "constant", "int"),
          "LatentDirichletAllocation__mean_change_tol": Parameter("LatentDirichletAllocation__mean_change_tol", [0, 0.1], "uniform", "float"),
          "LatentDirichletAllocation__n_jobs": Parameter("LatentDirichletAllocation__n_jobs", -1, "constant", "int"),
    }

    rules = []
    return LatentDirichletAllocation, sampler, rules


def get_configuration_MiniBatchDictionaryLearning():
    MiniBatchDictionaryLearning = ListTask(is_ordered=False, name = "MiniBatchDictionaryLearning",
                                  tasks = ["MiniBatchDictionaryLearning__n_components",
                                           "MiniBatchDictionaryLearning__alpha",
                                           "MiniBatchDictionaryLearning__fit_algorithm",
                                           "MiniBatchDictionaryLearning__transform_algorithm",
                                           "MiniBatchDictionaryLearning__transform_alpha",
                                           "MiniBatchDictionaryLearning__n_jobs",
                                           "MiniBatchDictionaryLearning__split_sign",
                                           "MiniBatchDictionaryLearning__batch_size",
                                           "MiniBatchDictionaryLearning__shuffle"])
    sampler = {
         "MiniBatchDictionaryLearning__n_components": Parameter("MiniBatchDictionaryLearning__n_components", [2, 20], 'uniform', "int"),
         "MiniBatchDictionaryLearning__alpha": Parameter("MiniBatchDictionaryLearning__alpha", [0, 1], "uniform", "float"),
         "MiniBatchDictionaryLearning__fit_algorithm": Parameter("MiniBatchDictionaryLearning__fit_algorithm", ["lars", "cd"], "choice", "string"),
         "MiniBatchDictionaryLearning__transform_algorithm": Parameter("MiniBatchDictionaryLearning__transform_algorithm", ["lasso_lars", "lasso_cd", "lars", "omp", "threshold"], "choice", "string"),
         "MiniBatchDictionaryLearning__transform_alpha": Parameter("MiniBatchDictionaryLearning__transform_alpha", [0, 1], "uniform", "float"),
         "MiniBatchDictionaryLearning__n_jobs": Parameter("MiniBatchDictionaryLearning__n_jobs", -1, "constant", "int"),
         "MiniBatchDictionaryLearning__split_sign": Parameter("MiniBatchDictionaryLearning__split_sign", [True, False], "choice", "bool"),
         "MiniBatchDictionaryLearning__batch_size": Parameter("MiniBatchDictionaryLearning__batch_size", [50, 400], "uniform", "int"),
         "MiniBatchDictionaryLearning__shuffle": Parameter("MiniBatchDictionaryLearning__shuffle", [True, False], "choice", "bool")
    }

    rules = []
    return MiniBatchDictionaryLearning, sampler, rules


def get_configuration_MiniBatchSparsePCA():
    MiniBatchSparsePCA = ListTask(is_ordered=False, name = "MiniBatchSparsePCA",
                                  tasks = ["MiniBatchSparsePCA__n_components",
                                           "MiniBatchSparsePCA__alpha",
                                           "MiniBatchSparsePCA__ridge_alpha",
                                           "MiniBatchSparsePCA__shuffle",
                                           "MiniBatchSparsePCA__batch_size",
                                           "MiniBatchSparsePCA__n_jobs",
                                           "MiniBatchSparsePCA__method"])
    sampler = {
          "MiniBatchSparsePCA__n_components": Parameter("MiniBatchSparsePCA__n_components", [2, 20], "uniform", 'int'),
          "MiniBatchSparsePCA__alpha": Parameter("MiniBatchSparsePCA__alpha", [0, 1], "uniform", 'float'),
          "MiniBatchSparsePCA__ridge_alpha": Parameter("MiniBatchSparsePCA__ridge_alpha", [0, 1], "uniform", 'float'),
          "MiniBatchSparsePCA__shuffle": Parameter("MiniBatchSparsePCA__shuffle", [True, False], "choice", "bool"),
          "MiniBatchSparsePCA__batch_size": Parameter("MiniBatchSparsePCA__batch_size", [10, 400], "uniform", "int"),
          "MiniBatchSparsePCA__n_jobs": Parameter("MiniBatchSparsePCA__n_jobs", -1, "constant", "int"),
          "MiniBatchSparsePCA__method": Parameter("MiniBatchSparsePCA__method", ["lars", "cd"], "choice", "string"),
    }

    rules = []
    return MiniBatchSparsePCA, sampler, rules


def get_configuration_NMF():
    NMF = ListTask(is_ordered=False, name = "NMF",
                                  tasks = ["NMF__n_components",
                                           "NMF__init",
                                           "NMF__solver",
                                           "NMF__beta_loss",
                                           "NMF__tol",
                                           "NMF__max_iter",
                                           "NMF__alpha",
                                           "NMF__l1_ratio",
                                           "NMF__shuffle"])
    sampler = {
          "NMF__n_components": Parameter("NMF__n_components", [2, 20], "uniform", 'int'),
          "NMF__init": Parameter("NMF__init", ["random", "nndsvd", "nndsvda", "nndsvdar"], "choice", 'string'),
          "NMF__solver": Parameter("NMF__solver", ["cd", "mu"], "choice", 'string'),
          "NMF__beta_loss": Parameter("NMF__beta_loss", ["frobenius", "kullback-leibler", "itakura-saito"], "choice", "string"),
          "NMF__tol": Parameter("NMF__tol", [0, 0.1], "uniform", "float"),
          "NMF__max_iter": Parameter("NMF__max_iter", [1, 100], "uniform", "int"),
          "NMF__alpha": Parameter("NMF__alpha", [0, 1], "uniform", "float"),
          "NMF__l1_ratio": Parameter("NMF__l1_ratio", [0, 1], "uniform", "float"),
          "NMF__shuffle": Parameter("NMF__shuffle", [True, False], "uniform", "bool"),
    }

    rules = [
        ChildRule(applied_to = ["NMF__beta_loss"], parent = "NMF__solver", value = ["mu"])
    ]
    return NMF, sampler, rules


def get_configuration_PCA():
    PCA = ListTask(is_ordered=False, name = "PCA",
                                  tasks = ["PCA__n_components",
                                           "PCA__whiten",
                                           "PCA__svd_solver",
                                           "PCA__tol",
                                           "PCA__iterated_power"])
    sampler = {
          "PCA__n_components": Parameter("PCA__n_components", [2, 20], "uniform", 'int'),
          "PCA__whiten": Parameter("PCA__whiten", [True, False], "choice", "bool"),
          "PCA__svd_solver": Parameter("PCA__svd_solver", ["auto", "full", "arpack", "randomized"], "choice", "string"),
          "PCA__tol": Parameter("PCA__tol", [0, 0.1], "uniform", "float"),
          "PCA__iterated_power": Parameter("PCA__iterated_power", "auto", "constant", "string"),
    }

    rules = [
        ChildRule(applied_to = ["PCA__iterated_power"], parent = "PCA__svd_solver", value = ["randomized"])
    ]
    return PCA, sampler, rules
