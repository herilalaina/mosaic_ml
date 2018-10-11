import unittest
import warnings

from sklearn import datasets, neural_network
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

from mosaic.mosaic import Search
from mosaic_ml.model_config.classification.neural_network import *

digits = datasets.load_digits()
X_digits = digits.data[:100]
y_digits = digits.target[:100]
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class TestNeuralNetwork(unittest.TestCase):

    def test_MLPClassifier(self):
        scenario, sampler, rules = get_configuration_MLPClassifier()

        def evaluate(config, bestconfig):
            for name, params in config:
                if  name == "MLPClassifier":
                    classifier = neural_network.MLPClassifier(**params)
                    classifier.fit(X_train, y_train)
                    return classifier.score(X_test, y_test)
            raise Exception("Classifier not found")

        searcher = Search(scenario, sampler, rules, evaluate)
        searcher.run(nb_simulation = 100, generate_image_path = "out/MLPClassifier")
