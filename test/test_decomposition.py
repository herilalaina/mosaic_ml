import unittest
import warnings

import pynisher
from mosaic.mosaic import Search
from mosaic_ml.model_config.data_preprocessing.decomposition import *
from sklearn import datasets, decomposition, linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=UserWarning)


@pynisher.enforce_limits(wall_time_in_s=20)
def fit_model(model, X, y):
    model.fit(X, y)
    return model


class TestDecomposition(unittest.TestCase):

    def test_DictionaryLearning(self):
        scenario, sampler, rules = get_configuration_DictionaryLearning()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "DictionaryLearning":
                        pipeline = Pipeline(steps=[(name, decomposition.DictionaryLearning(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/DictionaryLearning")

    def test_FactorAnalysis(self):
        scenario, sampler, rules = get_configuration_FactorAnalysis()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "FactorAnalysis":
                        pipeline = Pipeline(steps=[(name, decomposition.FactorAnalysis(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/FactorAnalysis")

    def test_FastICA(self):
        scenario, sampler, rules = get_configuration_FastICA()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "FastICA":
                        pipeline = Pipeline(steps=[(name, decomposition.FastICA(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/FastICA")

    def test_IncrementalPCA(self):
        scenario, sampler, rules = get_configuration_IncrementalPCA()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "IncrementalPCA":
                        pipeline = Pipeline(steps=[(name, decomposition.IncrementalPCA(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/IncrementalPCA")

    def test_KernelPCA(self):
        scenario, sampler, rules = get_configuration_KernelPCA()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "KernelPCA":
                        pipeline = Pipeline(steps=[(name, decomposition.KernelPCA(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        try:
                            pipeline = fit_model(pipeline, X_train, y_train)
                        except TimedOutExc:
                            return 0
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/KernelPCA")

    def test_LatentDirichletAllocation(self):
        scenario, sampler, rules = get_configuration_LatentDirichletAllocation()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "LatentDirichletAllocation":
                        pipeline = Pipeline(steps=[(name, decomposition.LatentDirichletAllocation(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/LatentDirichletAllocation")

    def test_MiniBatchDictionaryLearning(self):
        scenario, sampler, rules = get_configuration_MiniBatchDictionaryLearning()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "MiniBatchDictionaryLearning":
                        pipeline = Pipeline(steps=[(name, decomposition.MiniBatchDictionaryLearning(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/MiniBatchDictionaryLearning")

    def test_NMF(self):
        scenario, sampler, rules = get_configuration_NMF()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "NMF":
                        pipeline = Pipeline(steps=[(name, decomposition.NMF(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/NMF")

    def test_MiniBatchSparsePCA(self):
        scenario, sampler, rules = get_configuration_MiniBatchSparsePCA()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "MiniBatchSparsePCA":
                        pipeline = Pipeline(steps=[(name, decomposition.MiniBatchSparsePCA(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline.fit(X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/MiniBatchSparsePCA")

    def test_PCA(self):
        scenario, sampler, rules = get_configuration_PCA()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if name == "PCA":
                        pipeline = Pipeline(steps=[(name, decomposition.PCA(**params)),
                                                   ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation=10, generate_image_path="out/data_preprocessing/PCA")
