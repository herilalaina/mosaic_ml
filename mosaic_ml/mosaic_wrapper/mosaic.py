import logging
from mosaic_ml.mosaic_wrapper.mcts import MctsML
from mosaic.mosaic import Search


class SearchML(Search):
    """Main class to tune algorithm using Monte-Carlo Tree Search."""

    def __init__(self,
                 environment,
                 time_budget=3600,
                 seed=1,
                 policy_arg={},
                 exec_dir=""):
        """Initialization algorithm.

        :param environment: environment class extending AbstractEnvironment
        :param time_budget: overall time budget
        :param seed: random seed
        :param policy_arg: specific option for MCTS policy
        :param exec_dir: directory to store tmp files
        """
        super().__init__(environment, time_budget, seed, policy_arg, exec_dir)

        self.mcts = MctsML(env=environment,
                           time_budget=time_budget,
                           policy_arg=policy_arg,
                           exec_dir=exec_dir)

        # config logger for automl
        self.logger_automl = logging.getLogger('automl')

    def get_history_run(self):
        return self.mcts.env.final_model

    def test_performance(self, X_train, y_train, X_test, y_test, func_test, categorical_features):
        scores = []
        for r in self.mcts.env.final_model:
            time = r["running_time"]
            model = r["model"]
            try:
                score = func_test(model, X_train, y_train,
                                  X_test, y_test, categorical_features)
                if score is not None:
                    scores.append((time, score, r["cv_score"]))
            except Exception as e:
                print(e)
                pass
        return scores
