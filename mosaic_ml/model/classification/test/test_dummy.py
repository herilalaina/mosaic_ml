import unittest
import warnings

from numpy.linalg import LinAlgError

from sklearn import datasets, dummy
from sklearn.model_selection import train_test_split

from mosaic.mosaic import Search
from mosaic_ml.model.classification.dummy import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=UserWarning)

class TestDummy(unittest.TestCase):

    def test_DummyClassifier(self):
        scenario, sampler, rules = get_configuration_DummyClassifier()

        def evaluate(config):
            try:
                for name, params in config:
                    if  name == "DummyClassifier":
                        classifier = dummy.DummyClassifier(**params)
                        classifier.fit(X_train, y_train)
                        return classifier.score(X_test, y_test)
                raise Exception("Classifier not found")
            except LinAlgError:
                return 0

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 10, generate_image_path = "out/DummyClassifier")
