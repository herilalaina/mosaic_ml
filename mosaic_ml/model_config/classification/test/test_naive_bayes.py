import unittest

from sklearn import datasets, naive_bayes
from sklearn.model_selection import train_test_split

from mosaic.mosaic import Search
from mosaic_ml.model_config.classification.bernouilli_nb import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)


class TestNaiveBayes(unittest.TestCase):

    def test_GaussianNB(self):
        scenario, sampler, rules = get_configuration_GaussianNB()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "GaussianNB":
                    classifier = naive_bayes.GaussianNB(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/GaussianNB")

    def test_MultinomialNB(self):
        scenario, sampler, rules = get_configuration_MultinomialNB()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "MultinomialNB":
                    classifier = naive_bayes.MultinomialNB(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/MultinomialNB")
