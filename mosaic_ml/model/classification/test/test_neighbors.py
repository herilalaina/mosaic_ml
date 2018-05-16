import unittest

from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split

from mosaic.mosaic import Search
from mosaic_ml.model.classification.neighbors import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)


class TestNeighbors(unittest.TestCase):

    def test_KNeighborsClassifier(self):
        scenario, sampler, rules = get_configuration_KNeighborsClassifier()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "KNeighborsClassifier":
                    classifier = neighbors.KNeighborsClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/KNeighborsClassifier")

    def test_RadiusNeighborsClassifier(self):
        scenario, sampler, rules = get_configuration_RadiusNeighborsClassifier()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "RadiusNeighborsClassifier":
                    try:
                        classifier = neighbors.RadiusNeighborsClassifier(**params)
                        classifier.fit(X_train, y_train)
                        return classifier.score(X_test, y_test)
                    except:
                        return 0
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/RadiusNeighborsClassifier")
