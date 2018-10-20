import unittest
import warnings

import pynisher
from mosaic.mosaic import Search
from mosaic_ml.model_config.data_preprocessing.ensemble import *
from sklearn import datasets, ensemble, linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=UserWarning)

@pynisher.enforce_limits(wall_time_in_s=40)
def fit_model(model, X, y):
    model.fit(X, y)
    return model

class TestEnsemble(unittest.TestCase):

    def test_RandomTreesEmbedding(self):
        scenario, sampler, rules = get_configuration_RandomTreesEmbedding()

        def evaluate(config, bestconfig):
            try:
                for name, params in config:
                    if  name == "RandomTreesEmbedding":
                        pipeline = Pipeline(steps = [(name, ensemble.RandomTreesEmbedding(**params)), ("logistic_regression", linear_model.LogisticRegression())])
                        pipeline = fit_model(pipeline, X_train, y_train)
                        return pipeline.score(X_test, y_test)
                raise Exception("Classifier not found")
            except ValueError:
                return 0
            except pynisher.TimeoutException:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/data_preprocessing/RandomTreesEmbedding")
