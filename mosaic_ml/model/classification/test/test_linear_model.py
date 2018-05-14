import unittest
import warnings

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from mosaic.mosaic import Search
from mosaic_ml.model.classification.linear_model import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class TestLinearModel(unittest.TestCase):

    def test_LogisticRegression(self):
        scenario, sampler, rules = get_configuration_LogisticRegression()

        def evaluate(config):
            for name, params in config:
                if  name == "LogisticRegression":
                    classifier = linear_model.LogisticRegression(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/LogisticRegression")

    def test_SGDClassifier(self):
        scenario, sampler, rules = get_configuration_SGDClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "SGDClassifier":
                    classifier = linear_model.SGDClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/SGDClassifier")

    def test_RidgeClassifier(self):
        scenario, sampler, rules = get_configuration_RidgeClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "RidgeClassifier":
                    classifier = linear_model.RidgeClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/RidgeClassifier")

    def test_Perceptron(self):
        scenario, sampler, rules = get_configuration_Perceptron()

        def evaluate(config):
            for name, params in config:
                if  name == "Perceptron":
                    classifier = linear_model.Perceptron(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 5, generate_image_path = "out/Perceptron")

    def test_PassiveAggressiveClassifier(self):
        scenario, sampler, rules = get_configuration_PassiveAggressiveClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "PassiveAggressiveClassifier":
                    classifier = linear_model.PassiveAggressiveClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/PassiveAggressiveClassifier")
