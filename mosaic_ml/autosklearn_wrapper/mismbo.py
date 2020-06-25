# -*- encoding: utf-8 -*-

import time

from mosaic_ml.autosklearn_wrapper.metalearner import MetaLearningOptimizer

from autosklearn.constants import *

__all__ = [
    'calc_meta_features',
    'calc_meta_features_encoded',
    'convert_conf2smac_string',
    'create_metalearning_string_for_smac_call',
]


def suggest_via_metalearning(
        meta_base, dataset_name, metric, task, sparse,
        num_initial_configurations):

    if task == MULTILABEL_CLASSIFICATION:
        task = MULTICLASS_CLASSIFICATION

    task = TASK_TYPES_TO_STRING[task]

    start = time.time()
    ml = MetaLearningOptimizer(
        dataset_name=dataset_name,
        configuration_space=meta_base.configuration_space,
        meta_base=meta_base,
        distance='l1',
        seed=1,)
    runs = ml.metalearning_suggest_all(exclude_double_configurations=True)
    return runs[:num_initial_configurations]

def suggest_via_metalearning_(
        meta_base, dataset_name, metric, task, sparse,
        num_initial_configurations):

    if task == MULTILABEL_CLASSIFICATION:
        task = MULTICLASS_CLASSIFICATION

    task = TASK_TYPES_TO_STRING[task]

    start = time.time()
    ml = MetaLearningOptimizer(
        dataset_name=dataset_name,
        configuration_space=meta_base.configuration_space,
        meta_base=meta_base,
        distance='l1',
        seed=1,)
    runs = ml.metalearning_suggest_all_(exclude_double_configurations=True)
    return runs #[:num_initial_configurations]
