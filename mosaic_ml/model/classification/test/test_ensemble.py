import unittest
import warnings

from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from mosaic.mosaic import Search
from mosaic_ml.model.classification.ensemble import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class TestEnsemble(unittest.TestCase):

    def test_AdaBoostClassifier(self):
        scenario, sampler, rules = get_configuration_AdaBoostClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "AdaBoostClassifier":
                    classifier = ensemble.AdaBoostClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/AdaBoostClassifier")


    def test_BaggingClassifier(self):
        scenario, sampler, rules = get_configuration_BaggingClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "BaggingClassifier":
                    classifier = ensemble.BaggingClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 25, generate_image_path = "out/BaggingClassifier")

    def test_ExtraTreesClassifier(self):
        scenario, sampler, rules = get_configuration_ExtraTreesClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "ExtraTreesClassifier":
                    classifier = ensemble.ExtraTreesClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/ExtraTreesClassifier")

    def test_GradientBoostingClassifier(self):
        scenario, sampler, rules = get_configuration_GradientBoostingClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "GradientBoostingClassifier":
                    classifier = ensemble.GradientBoostingClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/ExtraTreesClassifier")


    def test_RandomForestClassifier(self):
        scenario, sampler, rules = get_configuration_RandomForestClassifier()

        def evaluate(config):
            for name, params in config:
                if  name == "RandomForestClassifier":
                    classifier = ensemble.RandomForestClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/RandomForestClassifier")
