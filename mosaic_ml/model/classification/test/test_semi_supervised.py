import unittest
import warnings

from sklearn import datasets, semi_supervised
from sklearn.model_selection import train_test_split

from mosaic.mosaic import Search
from mosaic_ml.model.classification.semi_supervised import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class TestSemiSuervised(unittest.TestCase):
    def test_LabelPropagation(self):
        scenario, sampler, rules = get_configuration_LabelPropagation()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "LabelPropagation":
                    classifier = semi_supervised.LabelPropagation(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/LabelPropagation")


    """def test_LabelSpreading(self):
        scenario, sampler, rules = get_configuration_LabelSpreading()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "LabelSpreading":
                    classifier = semi_supervised.LabelSpreading(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/LabelSpreading")"""
