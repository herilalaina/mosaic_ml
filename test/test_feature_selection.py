import unittest
import warnings

import pynisher
from mosaic.mosaic import Search
from mosaic_ml.model_config.data_preprocessing.feature_selection import *
from sklearn import datasets, linear_model, feature_selection
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@pynisher.enforce_limits(wall_time_in_s=20)
def fit_model(model, X, y):
    model.fit(X, y)
    return model

class Testfeature_selection(unittest.TestCase):

    def test_SelectPercentile(self):
        scenario, sampler, rules = get_configuration_SelectPercentile()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectPercentile":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectPercentile(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectPercentile")


    def test_SelectKBest(self):
        scenario, sampler, rules = get_configuration_SelectKBest()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectKBest":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectKBest(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectKBest")


    def test_SelectFpr(self):
        scenario, sampler, rules = get_configuration_SelectFpr()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectFpr":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectFpr(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectFpr")


    def test_SelectFdr(self):
        scenario, sampler, rules = get_configuration_SelectFdr()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectFdr":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectFdr(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectFdr")


    def test_SelectFwe(self):
        scenario, sampler, rules = get_configuration_SelectFwe()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectFwe":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectFwe(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        try:
                            pipeline = fit_model(pipeline, X_train, y_train)
                        except TimedOutExc:
                            return 0
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectFwe")


    def test_SelectFromModel(self):
        scenario, sampler, rules = get_configuration_SelectFromModel()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "SelectFromModel":
                        pipeline = Pipeline(steps = [(name, feature_selection.SelectFromModel(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/SelectFromModel")


    def test_RFE(self):
        scenario, sampler, rules = get_configuration_RFE()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "RFE":
                        pipeline = Pipeline(steps = [(name, feature_selection.RFE(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except Exception:
                return 0
            except TimedOutExc:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/RFE")
