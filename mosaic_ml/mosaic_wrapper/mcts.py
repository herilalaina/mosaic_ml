"""Monte carlo tree seach class."""

import logging
import os
import gc
import time
import numpy as np
import json

from mosaic.utils import Timeout
from mosaic.mcts import MCTS
from networkx.readwrite.gpickle import write_gpickle


class MctsML(MCTS):
    """Monte carlo tree search implementation."""

    def __init__(self, env,
                 time_budget=3600,
                 bandit_policy = None,
                 exec_dir = ""):
        super().__init__(env=env, bandit_policy=bandit_policy, time_budget=time_budget, exec_dir=exec_dir, coef_progressive_widening=0.6)

    def MCT_SEARCH(self):
        reward, config = super().MCT_SEARCH()

        write_gpickle(self.tree, os.path.join(self.exec_dir, "tree.pkl"))
        with open(os.path.join(self.exec_dir, "full_log.json"), 'w') as outfile:
            json.dump(self.env.history_score, outfile)

        return reward, config

    def create_node_for_algorithm(self):
        id_class = {}
        for cl in ["bernoulli_nb", "multinomial_nb",
                    "decision_tree", "gaussian_nb", "sgd",
                    "passive_aggressive", "xgradient_boosting",
                    "adaboost", "extra_trees", "gradient_boosting",
                    "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors", "random_forest"]:
            id_class[cl] = self.tree.add_node(name="classifier:__choice__", value=cl, terminal=False, parent_node = 0)
        return id_class

    def run(self, n = 1, initial_configurations = [], nb_iter_to_generate_img = ""):
        start_run = time.time()

        with Timeout(int(self.time_budget - (start_run - time.time()))):
            try:
                self.logger.info("Run default configuration")
                self.bestconfig, self.bestscore = self.env.run_default_configuration()
                self.env.check_time()

                if len(initial_configurations) > 0:
                    self.logger.info("Run intial configurations")
                    executed_config = self.env.run_default_all()
                    id_class = self.create_node_for_algorithm()
                    score_each_cl = self.env.run_initial_configuration(initial_configurations, executed_config)
                    for cl, vals in score_each_cl.items():
                        if len(vals) > 0:
                            [self.BACKUP(id_class[cl], s) for s in vals]
                else:
                    self.logger.info("No initial configuration to run.")
                    self.env.check_time()
                    self.env.run_main_configuration()

                for i in range(n):
                    if time.time() - self.env.start_time < self.time_budget:
                        res, config = self.MCT_SEARCH()

                        if res > self.bestscore:
                            self.bestscore = res
                            self.bestconfig = config
                    else:
                        self.logger.info("Budget exhausted.")
                        return 0

                    if nb_iter_to_generate_img == -1 or i % nb_iter_to_generate_img == 0:
                        self.tree.draw_tree(
                            os.path.join(self.exec_dir, "images"))
            except Timeout.Timeout:
                self.logger.info("Budget exhausted.")
                return 0

    def print_tree(self, images):
        self.tree.draw_tree(images)
