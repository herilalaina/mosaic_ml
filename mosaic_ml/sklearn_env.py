"""Base environement class."""

import pynisher
import numpy as np
import time
import logging
from datetime import datetime

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, IntegerHyperparameter
from mosaic.external.ConfigSpace.util import get_one_exchange_neighbourhood_with_history
from mosaic.external.ConfigSpace.configuration_space import Configuration
from mosaic.utils import Timeout, get_index_percentile

from mosaic.env import MosaicEnvironment
from sklearn.metrics.pairwise import cosine_similarity

from mosaic.utils import expected_improvement
from mosaic_ml.model_score import ScoreModel


class SklearnEnv(MosaicEnvironment):
    """Base class for environement."""

    def __init__(self, eval_func,
                 config_space,
                 mem_in_mb=3024,
                 cpu_time_in_s=300,
                 use_parameter_importance=True,
                 seed = 1):
        """Constructor."""
        super().__init__(seed)
        self.eval_func = eval_func
        self.config_space = config_space
        self.mem_in_mb = mem_in_mb
        self.cpu_time_in_s = cpu_time_in_s

        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()
        self.use_parameter_importance = use_parameter_importance

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.score_model = ScoreModel(len(self.config_space._hyperparameters),
                                        id_most_import_class=[self.config_space.get_idx_by_hyperparameter_name(p) for p in ["categorical_encoding:__choice__", "classifier:__choice__"]])
        self.history_score = []
        self.logger = logging.getLogger('mcts')
        self.final_model = []

        # statistics
        self.sucess_run = 0
        self.rng = np.random.RandomState(seed)

        self.id = 0
        self.main_hyperparameter = ["classifier:__choice__", "preprocessor:__choice__", "categorical_encoding:__choice__", "imputation:strategy", "rescaling:__choice__"]
        self.max_nb_child_main_parameter = {
            "root": 16,
            "classifier:__choice__:adaboost": 10,
            "classifier:__choice__:bernoulli_nb": 13,
            "classifier:__choice__:decision_tree": 10,
            "classifier:__choice__:extra_trees": 10,
            "classifier:__choice__:gaussian_nb": 9,
            "classifier:__choice__:gradient_boosting": 9,
            "classifier:__choice__:k_nearest_neighbors": 10,
            "classifier:__choice__:lda": 12,
            "classifier:__choice__:liblinear_svc": 13,
            "classifier:__choice__:libsvm_svc": 10,
            "classifier:__choice__:multinomial_nb": 8,
            "classifier:__choice__:passive_aggressive": 13,
            "classifier:__choice__:qda": 12,
            "classifier:__choice__:random_forest": 10,
            "classifier:__choice__:sgd": 13,
            "classifier:__choice__:xgradient_boosting": 10,
            "preprocessor:__choice__": 2,
            "categorical_encoding:__choice__": 3,
            "imputation:strategy": 6,
            "rescaling:__choice__": 0
        }
        self.nb_choice_hyp = self._get_nb_choice_for_each_parameter()

        self.nb_exec_for_params = dict()
        for p in self.config_space.get_hyperparameter_names():
            self.nb_exec_for_params[p] = {"nb": 0, "ens": set()}

        self.current_rollout = None

        self.experts = {}

        self.problem_dependant_param = []
        self.problem_dependant_value = {}

    def print_config(self):
        self.logger_automl.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.logger_automl.info("Memory limit = {0} MB".format(self.mcts.env.mem_in_mb))
        self.logger_automl.info("Overall Time Budget = {0}".format(self.mcts.time_budget))
        self.logger_automl.info("Evaluation Time Limit = {0}".format(
            self.mcts.env.cpu_time_in_s))
        self.logger_automl.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def reset(self, eval_func,
              mem_in_mb=3024,
              cpu_time_in_s=30):
        self.bestconfig = {
            "validation_score": 0,
            "model": None
        }
        self.start_time = time.time()

        # Constrained evaluation
        self.max_eval_time = cpu_time_in_s
        self.cpu_time_in_s = cpu_time_in_s
        self.mem_in_mb = mem_in_mb
        self.eval_func = eval_func

        self.history_score = []
        self.current_rollout = None

    def rollout(self, history = []):
        while True:
            try:
                return self.config_space.sample_partial_configuration_with_default(history)
            except Exception as e:
                pass

    def fix_valid_configuration(self, config_):
        try:
            config = config_
            arr = config_.get_dictionary()
            for p, p_problem in zip([117, 120, 122, 132], self.problem_dependant_param):
                if p_problem in config:
                    arr[p_problem] = min([config[p_problem], 10])


            if not self.problem_dependant_value["is_positive"]:
                for p in ["preprocessor:select_percentile_classification:score_func", "preprocessor:select_rates:score_func"]:
                    if p in arr:
                        arr[p] = "f_classif"


            return Configuration(configuration_space=self.config_space, values=arr, allow_inactive_with_values=True)
        except:
            pass
        return config_

    def fix_rollout_value(self, configs):
        try:
            # Encoding
            mask_normal_encoding = configs[:, 1] == 0
            mask_onehot_encoding = configs[:, 1] == 1
            for p, name_p in zip([117, 120, 122, 132], self.problem_dependant_param):
                configs[mask_normal_encoding, p] = np.minimum(configs[mask_normal_encoding, p], self.config_space.get_hyperparameter(name_p)._transform(10))
                configs[mask_onehot_encoding, p] = np.minimum(configs[mask_onehot_encoding, p], self.config_space.get_hyperparameter(name_p)._transform(10))

            if not self.problem_dependant_value["is_positive"]:
                configs[configs[:, 146] == 0, 146] = 1
                configs[configs[:, 149] == 0, 149] = 1

            return configs
        except Exception as e:
            print(e)
        return configs

    def is_metalearning(self):
        return self.score_model.active_general_model

    def rollout_in_expert_neighborhood(self, history = []):
        try:
            expert, _ = self.experts[tuple(history)]
        except:
            expert = self.config_space.sample_partial_configuration_with_default(history)

        try:
            #configs = [c_f for c_f in
            #    [get_one_exchange_neighbourhood_with_history(c_i, self.seed, history) for c_i in
            #        [c for c in get_one_exchange_neighbourhood_with_history(expert, self.seed, history)]
            #]]
            configs = []
            # print("[Rollout] Neighborhood on", expert.get_dictionary())
            for c in get_one_exchange_neighbourhood_with_history(expert, self.seed, history):
                tmp_ = list(get_one_exchange_neighbourhood_with_history(c, self.seed, history))
                configs.extend(tmp_)
            configs.extend(self.config_space.sample_partial_configuration(history, 1000))

            list_configuration_to_choose = np.nan_to_num(np.array([c.get_array() for c in configs]))

        except Exception as e:
            raise(e)

        mu_loc, sigma_loc = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), "local")
        ei_values = expected_improvement(mu_loc, sigma_loc, self.bestconfig["validation_score"])
        #mu_time, sigma_time = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), "time")

        #if self.is_metalearning:
        #    beta = self.get_beta()
        #    mu_gen, sigma_gen = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), "general")

        #ei_values = np.array([mu_gener * beta + (1 - beta) * mu_local for mu_gener, mu_local in zip(mu, mu_loc)])

        id_max = (-ei_values).argsort()[:3]
        return [configs[np.random.choice(id_max)] for _ in range(3)]


    def rollout_with_model_performance(self, history=[]):
        value_to_choose = []
        list_configuration_to_choose = []
        list_rollout = []
        st_time=time.time()
        try:
            configs = self.config_space.sample_partial_configuration(history, 500)
            list_configuration_to_choose = [np.nan_to_num(c.get_array()) for c in configs]
        except Exception as e:
            pass
        mu, sigma = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model)
        ei_values = expected_improvement(mu, sigma, self.bestconfig["validation_score"])
        mu_time, sigma_time = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), self.score_model.model_of_time)
        ei_values = [ei if mu_t < (self.cpu_time_in_s) else -1000 for ei, mu_t in zip(ei_values, mu_time)]

        id_max = np.argmax(ei_values)
        return configs[id_max]

    def get_beta(self):
        return (time.time() - self.start_time) / 3600

    def _has_multiple_value(self, p):
        next_param_cs = self.config_space._hyperparameters[p]
        ens_val = set()
        for _ in range(5):
            ens_val.add(next_param_cs.sample(self.rng))
        return len(ens_val) > 1

    def _can_use_parameter_importance(self, list_params, threshold = 10):
        for p in list_params:
            if self.nb_exec_for_params[p]["nb"] < 10 or (self._has_multiple_value(p) and len(self.nb_exec_for_params[p]["ens"]) == 1):
                return False
        return True

    def importance_surgate(self, param, history):
        list_configuration_to_choose = []
        value_to_choose = set()
        next_param_cs = self.config_space._hyperparameters[param]
        for _ in range(1000):
            next_param_v = next_param_cs.sample(self.rng)
            try:
                ex_config = self.config_space.sample_partial_configuration(history + [(param, next_param_v)])
                vect_config = np.nan_to_num(ex_config.get_array())
                if len(vect_config) == 172 and next_param_v not in value_to_choose:
                    list_configuration_to_choose.append(vect_config)
                    value_to_choose.append(next_param_v)
            except Exception as e:
                pass
        mu, sigma = self.score_model.get_mu_sigma_from_rf(np.array(list_configuration_to_choose), "local")
        return np.mean(mu)

    def can_be_selectioned(self, possible_params, child_info, history):
        history_ens = [v[0] for v in history]
        params_to_check = set(history_ens).intersection(set(possible_params))
        if len(params_to_check) > 0:
            final_available_param = possible_params
            for p in params_to_check:
                possible_params.remove(p)
                buffer = set()
                p_cs = self.config_space.get_hyperparameter(p)
                for new_val in p_cs.choices:
                    try:
                        self.config_space.sample_partial_configuration(history + [(p, new_val)])
                        buffer.add(new_val)
                    except Exception as e:
                        pass

                if len([param for param in child_info if param == p]) < len(buffer):
                    possible_params.append(p)

        return possible_params



    def next_move(self, history=[], info_children=[]):
        st_time = time.time()
        try:
            #config = self.config_space.sample_partial_configuration(history)

            possible_params_ = list(self.config_space.get_possible_next_params(history))
            possible_params_ = list(set(possible_params_).intersection(set(self.main_hyperparameter)))
            possible_params = self.can_be_selectioned(possible_params_, [v[0] for v in info_children], history)
            #print("Clean possible parameter ", time.time() - st_time)

            if set(self.main_hyperparameter).intersection(set(possible_params)):
                for main_p in self.main_hyperparameter:
                    if main_p in possible_params:
                        id_param = possible_params.index(main_p)
                        break
            elif self.use_parameter_importance and self._can_use_parameter_importance(possible_params):
                # print("Parameter importance activated")
                id_param = self.score_model.most_importance_parameter([self.config_space.get_idx_by_hyperparameter_name(p) for p in possible_params])
            else:
                id_param = np.random.randint(0, len(possible_params))
            next_param = possible_params[id_param]

            next_param_cs = self.config_space._hyperparameters[next_param]

            buffer_config = []
            value_to_choose = []

            if isinstance(next_param_cs, CategoricalHyperparameter):
                list_choice = next_param_cs.choices
            elif isinstance(next_param_cs, IntegerHyperparameter):
                list_choice = [next_param_cs.sample(self.rng) for _ in range(10)]

            for next_param_v in list_choice:
                stt_time = time.time()
                if next_param_v not in value_to_choose and next_param_v not in [v[1] for v in info_children if v[0] == next_param]:
                    tmp_config = []
                    try:
                        ex_config = self.config_space.sample_partial_configuration(history + [(next_param, next_param_v)], 500)
                        tmp_config = np.nan_to_num([c.get_array() for c in ex_config])
                    except Exception as e:
                        # print("Can not sample for ", next_param, next_param_v)
                        pass
                    if len(tmp_config) > 0:
                        value_to_choose.append(next_param_v)
                        buffer_config.append(tmp_config)
            try:
                list_value_score = []
                for list_config in buffer_config:
                    beta = self.get_beta()
                    mu_loc, _ = self.score_model.get_mu_sigma_from_rf(np.array(list_config), "local")
                    list_value_score.append(np.median(mu_loc))

                id_max = np.argmax(list_value_score)
                value_param = value_to_choose[id_max]
            except Exception as e:
                # print("error in ei")
                raise(e)
                value_param = np.random.choice(value_to_choose)

            history.append((next_param, value_param))
        except Exception as e:
            # print("Exception for {0}".format(history))
            # print("Child info", info_children)
            # print("Possible params", possible_params)
            # print("Next params", next_param)
            # print("Value to choose", value_to_choose)
            # print("Value example: ", next_param_v)
            raise (e)

        # print("Current pipeline: ", history)
        # print("Possible params", possible_params)
        # print("Next params", (next_param, value_param))
        #is_terminal = not self._check_if_same_pipeline([el[0] for el in history], [el for el in config])
        possible_params.remove(next_param)
        is_terminal = len(set(possible_params).intersection(set(self.main_hyperparameter))) == 0
        return next_param, value_param, is_terminal



    def _valid_sample(self, history, config):
        for hp_name, hp_value in history:
            if hp_name not in config.keys() or config[hp_name] != hp_value:
                return False
        return True

    def _preprocess_moves_util(self, list_moves, index):
        model = list_moves[index][0]
        param_model = {}
        last_index = index
        for i in range(index + 1, len(list_moves)):
            last_index = i
            config, val = list_moves[i]
            if config.startswith(model):
                param_model[config.split("__")[1]] = val
            else:
                break
        return model, param_model, last_index

    def _preprocess_moves(self, list_moves):
        index = 0
        preprocessed_moves = []
        while (index < len(list_moves) - 1):
            model, params, index = self._preprocess_moves_util(list_moves, index)
            preprocessed_moves.append((model, params))
        return preprocessed_moves

    def evaluate(self, config):
        return self._evaluate(config)

    def _evaluate(self, config, type="normal"):
        self.check_time()
        self.id += 1
        start_time = time.time()
        if type == "normal":
            eval_func = pynisher.enforce_limits(mem_in_mb=self.mem_in_mb, cpu_time_in_s=self.cpu_time_in_s)(self.eval_func)
        elif type == "init":
            eval_func = pynisher.enforce_limits(mem_in_mb=5000, cpu_time_in_s=self.cpu_time_in_s)(self.eval_func)
        else:
            eval_func = self.eval_func
        try:
            res = eval_func(config, self.bestconfig, self.id)
            self.sucess_run += 1

        except Timeout.Timeout as e:
            print(e)
            res = None
            raise(e)

        if res is None:
            res = {"validation_score": 0, "info": None}

        res["running_time"] = time.time() - start_time

        self._update_expert(config, res["validation_score"])

        if type != "default":
            res["predict_performance"] = self.score_model.get_performance(np.nan_to_num(config.get_array()))

        if res["validation_score"] > 0:
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), res["validation_score"], res["running_time"])
        else:
            self.score_model.partial_fit(np.nan_to_num(config.get_array()), 0, 3000)

        self.log_result(res, config)

        return res["validation_score"]

    def run_default_configuration(self):
        # print("Run default configuration")
        try:
            config = self.config_space.get_default_configuration()
            return config, self._evaluate(config, type="default")
        except Exception as e:
            raise(e)

    def check_time(self):
        if time.time() - self.start_time < 3600:
            return True
        else:
            raise Timeout.Timeout("Finished")


    def run_default_all(self):
        # print("Run default configuration")

        set_config = set()
        for cl in ["bernoulli_nb", "multinomial_nb", "decision_tree", "gaussian_nb", "sgd", "passive_aggressive", "xgradient_boosting", "adaboost", "extra_trees", "gradient_boosting", "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors"]:
            config = self.config_space.sample_partial_configuration_with_default([("classifier:__choice__", cl)])
            self.check_time()
            score = self._evaluate(config)
            set_config.add(config)
        return set_config



    def run_main_configuration(self):
        # print("Run main configuration")
        id_score = {}
        for cl in ["bernoulli_nb", "multinomial_nb", "decision_tree", "gaussian_nb", "sgd", "passive_aggressive", "xgradient_boosting", "adaboost", "extra_trees", "gradient_boosting", "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors", "random_forest"]:
            id_score[cl] = []

        for cl in ["bernoulli_nb", "multinomial_nb", "decision_tree", "gaussian_nb", "sgd", "passive_aggressive", "xgradient_boosting", "adaboost", "extra_trees", "gradient_boosting", "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors", "random_forest"]:
            yield self.config_space.sample_partial_configuration_with_default([("classifier:__choice__", cl)])
            # st_time = time.time()
            # score = self._evaluate(config)
            # id_score[config["classifier:__choice__"]].append(score)
            """if time.time() - st_time > 50:
                continue"""
            for _ in range(1):
                # self.check_time()
                yield self.config_space.sample_partial_configuration([("classifier:__choice__", cl)])
                # yield config

        #score = self._evaluate(config)

        #id_score[config["classifier:__choice__"]].append(score)
        #return id_score

    def run_random_configuration(self):
        # print("Run random configuration")
        self._evaluate(self.config_space.sample_configuration())

    def run_initial_configuration(self, intial_configuration, already_executed = set({})):
        # print("Run initial configuration")
        id_score = {}
        for cl in ["bernoulli_nb", "multinomial_nb", "decision_tree", "gaussian_nb", "sgd", "passive_aggressive", "xgradient_boosting", "adaboost", "extra_trees", "gradient_boosting", "lda", "liblinear_svc", "libsvm_svc", "qda", "k_nearest_neighbors", "random_forest"]:
            id_score[cl] = []
        for c in intial_configuration:
            if c not in already_executed:
                self.check_time()
                score = self._evaluate(c)
                id_score[c["classifier:__choice__"]].append(score)

        return id_score


    def _get_nb_choice_for_each_parameter(self):
        count_dict = {}
        for hyp in self.config_space.get_hyperparameter_names():
            hyp_cs = self.config_space.get_hyperparameter(hyp)
            if isinstance(hyp_cs, CategoricalHyperparameter):
                count_dict[hyp] = len(hyp_cs.choices)
            elif isinstance(hyp_cs, IntegerHyperparameter):
                buffer = set()
                for _ in range(100):
                    buffer.add(hyp_cs.sample(self.rng))
                count_dict[hyp] = len(buffer)
            else:
                count_dict[hyp] = 100
        return count_dict

    def get_nb_children(self, parameter, value, current_pipeline):
        """Get the number of

        :param parameter:
        :param value:
        :return: maximum number of children
        """
        if parameter.startswith("classifier:__choice__"):
            max_number_of_child = self.max_nb_child_main_parameter[parameter + ":" + value]
        elif parameter in self.max_nb_child_main_parameter:
            max_number_of_child = self.max_nb_child_main_parameter[parameter]
        else:
            max_number_of_child = 20

        return max_number_of_child

    def add_to_final_model(self, config):
        self.final_model.append(config)

    def _update_expert(self, config, score):
        buffer = []
        for params in self.main_hyperparameter:
            buffer.append((params, config.get(params)))

            if tuple(buffer) in self.experts:
                expert_config, expert_score = self.experts[tuple(buffer)]
                if expert_score < score:
                    self.experts[tuple(buffer)] = (config, score)
            else:
                self.experts[tuple(buffer)] = (config, score)

    def estimate_action_state(self, state, next_state, action, local_model = True):
        if local_model == "general" and not self.is_metalearning:
            local_model = "local"
        configs = self.config_space.sample_partial_configuration(state + [(next_state, action)], 1000)
        mu, _ = self.score_model.get_mu_sigma_from_rf(np.nan_to_num([c.get_array() for c in configs]), "local" if local_model else "general")
        return np.mean(mu)

    def get_nearest_data(self, id):
        ids = []
        ranks = []
        for f_ in glob.glob("/scratch/hrakotoa/mosaic_metadata/metadata_*"):
            with open(f_, 'r') as f:
                if "metadata_mosaic_" not in f_:
                    id_data = int(f_.split("metadata_")[1])
                    if id_data not in [3021, 3573, 3946, 3948, 14967]:
                        reader = csv.reader(f)
                        for row in reader:
                            ranks.append([ float(i) for i in row ])
                            ids.append(id_data)
        ranks = np.array(ranks)
        X_sim = cosine_similarity(ranks)

        if id in ids:
            self.score_model.dataset_features = ranks[ids.index(id)]
        else:
            return list(np.random.choice(ids, 5))

        sim_for_data = X_sim[ids.index(id)]

        id_nears = sim_for_data.argsort()[-6:-1][::-1]
        return [ids[id_near] for id_near in id_nears], [ranks[id_near] for id_near in id_nears]

    def get_metadata_mosaic(self, id):
        path_to_dump = os.path.join("/scratch/hrakotoa/mosaic_metadata/", "metadata_mosaic_{0}".format(id))
        data = json.load(open(path_to_dump, "r"))
        return data

    def get_test_performance_neighbors(self, id):
        id_neighborhoods, features = self.get_nearest_data(id)
        X = []
        Y = []
        for i_, id_n in enumerate(id_neighborhoods):
            for seed in range(1, 7):
                try:
                    path = "/scratch/hrakotoa/mosaic_exec/{0}/{1}/".format(seed, id_n)
                    X_ = np.load(os.path.join(path, "X.npy"))
                    Y_ = np.load(os.path.join(path, "y.npy"))
                    X.extend([np.concatenate([x, features[i_]]) for x in X_])
                    Y.extend(Y_)
                except Exception as e:
                    # print(e)
                    pass
        return X, Y

    def load_metalearning_x_y(self, id):
        #X, y = self.get_test_performance_neighbors(id)
        #self.score_model.active_general_model = True
        #self.score_model.model_general.fit(X, y)
        id_neighborhoods, features = self.get_nearest_data(id)

        configs = []

        list_config = []
        config_set = set()
        for i_, id_n in enumerate(id_neighborhoods):
            list_config.append(self.get_metadata_mosaic(id_n))
        for cl in ["random_forest", "xgradient_boosting", "adaboost", "extra_trees", "libsvm_svc"]:
            for c in list_config:
                try:
                    configuration = Configuration(configuration_space=self.config_space, values=c[cl], allow_inactive_with_values=True)
                    if configuration not in config_set:
                        config_set.add(configuration)
                        configs.append(configuration)

                except Exception as e:
                    # print(e)
                    pass
        return configs

    def log_result(self, res, config):
        run = res
        run["id"] = self.id
        run["elapsed_time"] = time.time() - self.start_time
        run["model"] = config.get_dictionary()
        for k, v in config.get_dictionary().items():
            self.nb_exec_for_params[k]["nb"] = self.nb_exec_for_params[k]["nb"] + 1
            self.nb_exec_for_params[k]["ens"].add(v)

        self.history_score.append(run)

        # print(">> {0}: validation score: {1}\n".format(str(config), res["validation_score"]))

        if res["validation_score"] > self.bestconfig["validation_score"]:
            self.add_to_final_model(run)
            self.bestconfig = run
