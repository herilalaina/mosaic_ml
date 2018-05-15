import random

from mosaic.scenario import ListTask, ComplexScenario, ChoiceScenario
from mosaic.mosaic import Search

from mosaic_ml import model
from mosaic_ml.utils import time_limit

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from functools import partial

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
        self.start = ComplexScenario(name = "root", scenarios=[preprocessing, classifier_model], is_ordered=True)
        self.adjust_sampler()

    def adjust_sampler(self):
        for param in model.DATA_DEPENDANT_PARAMS:
            if param in self.sampler:
                self.sampler[param].value_list = [1, self.X.shape[1] - 1]


    def fit(self, X, y):
        self.X = X
        self.y = y
        self.configure_hyperparameter_space()

        def evaluate(config, X_train=None, X_test=None, y_train=None, y_test=None):
            print("\n#####################################################")
            preprocessing = None
            classifier = None

            for name, params in config:
                if name in model.list_available_preprocessing:
                    preprocessing = model.list_available_preprocessing[name](**params)
                if  name in model.list_available_classifiers:
                    classifier = model.list_available_classifiers[name](**params)

            if preprocessing is None or classifier is None:
                raise Exception("Classifier and/or Preprocessing not found\n {0}".format(config))

            pipeline = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])

            try:
                a = 20
                print(pipeline)
                with time_limit(a):
                    pipeline.fit(X, y)
                score = pipeline.score(X_test, y_test)
                print(">>>>>>>>>>>>>>>> Score: {0}".format(score))
                return score
            except Exception as e:
                print(e)
                return 0

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        eval_func = partial(evaluate, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        self.searcher = Search(self.start, self.sampler, self.rules, eval_func)
        self.searcher.run(nb_simulation = 500, generate_image_path = "search_out")
