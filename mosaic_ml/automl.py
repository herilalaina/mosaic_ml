import random

from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario
from mosaic.mosaic import Search
from mosaic_ml import model

LIST_TASK = [""]

class AutoML():
    def __init__(self, time_budget = None, time_limit_for_evaluation = None):
        self.time_budget = time_budget
        self.time_limit_for_evaluation = time_limit_for_evaluation

    def configure_hyperparameter_space(self):
        if not hasattr(self, "X") or not hasattr(self, "y"):
            raise Exception("X, y is not defined.")

        # Data preprocessing
        list_data_preprocessing = model.get_all_data_preprocessing()
        data_preprocessing = []
        self.rules = []
        self.sampler = {}
        for dp_method in list_data_preprocessing:
            scenario, sampler_, rules_ = dp_method()
            data_preprocessing.append(scenario)
            self.rules.extend(rules_)
            self.sampler.update(sampler_)
        preprocessing = ChoiceScenario(name = "preprocessing", scenarios=data_preprocessing)

        # Classifier
        list_classifiers = model.get_all_classifier()
        classifiers = []
        for clf in list_classifiers:
            scenario, sampler_, rules_ = clf()
            classifiers.append(scenario)
            self.rules.extend(rules_)
            self.sampler.update(sampler_)
        classifier_model = ChoiceScenario(name = "classifier", scenarios=classifiers)

        # Pipeline = preprocessing + classifier
        start = ComplexScenario(name = "root", scenarios=[preprocessing, classifier_model], is_ordered=True)


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.configure_hyperparameter_space()

        def evaluate(config):
            print(config)
            return random.uniform(0, 1)

        self.searcher = Search(self.start, self.sampler, self.rules, evaluate)
        self.searcher.run(nb_simulation = 10000, generate_image_path = "search_out")
