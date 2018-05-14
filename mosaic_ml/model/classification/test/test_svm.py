import unittest

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

from mosaic.mosaic import Search
from mosaic_ml.model.classification.svm import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)


class TestSVM(unittest.TestCase):
    def test_LinearSVC(self):
        scenario, sampler, rules = get_configuration_LinearSVC()

        def evaluate(config):
            for name, params in config:
                if  name == "LinearSVC":
                    classifier = svm.LinearSVC(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 50, generate_image_path = "out/LinearSVC")
