import unittest
import warnings

from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from mosaic.mosaic import Search
from mosaic_ml.model.classification.tree import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


class TestDecisionTreeClassifier(unittest.TestCase):

    def test_DecisionTreeClassifier(self):
        scenario, sampler, rules = get_configuration_DecisionTreeClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "DecisionTreeClassifier":
                    classifier = tree.DecisionTreeClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/DecisionTreeClassifier")

    def test_ExtraTreeClassifier(self):
        scenario, sampler, rules = get_configuration_ExtraTreeClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "ExtraTreeClassifier":
                    classifier = tree.ExtraTreeClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/ExtraTreeClassifier")
