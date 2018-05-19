import unittest
import warnings
import pynisher

from sklearn import datasets, preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mosaic.mosaic import Search
from mosaic_ml.model.data_preprocessing.preprocessing import *

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=UserWarning)

@pynisher.enforce_limits(wall_time_in_s=40)
def fit_model(model, X, y):
    model.fit(X, y)
    return model

class TestPreprocessing(unittest.TestCase):

    def test_PolynomialFeatures(self):
        scenario, sampler, rules = get_configuration_PolynomialFeatures()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "PolynomialFeatures":
                        pipeline = Pipeline(steps = [(name, preprocessing.PolynomialFeatures(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        if pipeline is not None:
                            return pipeline.score(X_test, y_test)
                        else:
                            return 0
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except pynisher.TimeoutException:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/PolynomialFeatures")

    def test_FunctionTransformer(self):
        scenario, sampler, rules = get_configuration_FunctionTransformer()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "FunctionTransformer":
                        pipeline = Pipeline(steps = [(name, preprocessing.FunctionTransformer(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except pynisher.TimeoutException:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/FunctionTransformer")
