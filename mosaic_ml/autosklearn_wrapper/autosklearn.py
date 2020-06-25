#! /usr/bin/python -u

# Below source code is extracted from https://github.com/automl/auto-sklearn

import os
import scipy

import mosaic_ml
import numpy as np
from autosklearn.util import pipeline

from autosklearn import metalearning
from autosklearn.smbo import EXCLUDE_META_FEATURES_CLASSIFICATION, EXCLUDE_META_FEATURES_REGRESSION, CLASSIFICATION_TASKS, \
                        MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION, MULTILABEL_CLASSIFICATION, REGRESSION
from autosklearn.constants import *
from mosaic_ml.autosklearn_wrapper.mismbo import suggest_via_metalearning_
from autosklearn.metalearning.metalearning.meta_base import MetaBase

from autosklearn.data.abstract_data_manager import perform_one_hot_encoding
from autosklearn.metalearning.metafeatures.metafeatures import \
    calculate_all_metafeatures_with_labels, calculate_all_metafeatures_encoded_labels

def _calculate_metafeatures__(data_feat_type, data_info_task, basename,
                            x_train, y_train):
    # == Calculate metafeatures
    task_name = 'CalculateMetafeatures'
    categorical = [True if feat_type.lower() in ['categorical'] else False
                   for feat_type in data_feat_type]

    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if data_info_task in CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

    if data_info_task in [MULTICLASS_CLASSIFICATION, BINARY_CLASSIFICATION,
                          MULTILABEL_CLASSIFICATION, REGRESSION]:
        result = calculate_all_metafeatures_with_labels(
            x_train, y_train, categorical=categorical,
            dataset_name=basename,
            dont_calculate=EXCLUDE_META_FEATURES, )
        for key in list(result.metafeature_values.keys()):
            if result.metafeature_values[key].type_ != 'METAFEATURE':
                del result.metafeature_values[key]
    else:
        result = None
    return result

def _calculate_metafeatures_encoded__(basename, x_train, y_train):
    EXCLUDE_META_FEATURES = EXCLUDE_META_FEATURES_CLASSIFICATION \
        if task in CLASSIFICATION_TASKS else EXCLUDE_META_FEATURES_REGRESSION

    task_name = 'CalculateMetafeaturesEncoded'
    result = calculate_all_metafeatures_encoded_labels(
        x_train, y_train, categorical=[False] * x_train.shape[1],
        dataset_name=basename, dont_calculate=EXCLUDE_META_FEATURES)
    for key in list(result.metafeature_values.keys()):
        if result.metafeature_values[key].type_ != 'METAFEATURE':
            del result.metafeature_values[key]

    return result



def get_autosklearn_metalearning(X_train, y_train, cat, metric, num_initial_configurations):
    task_id = "new_task"
    is_sparse = scipy.sparse.issparse(X_train)

    dataset_properties = {
        'signed': True,
        'multiclass': False if len(np.unique(y_train)) == 2 == 2 else True,
        'task': 1 if len(np.unique(y_train)) == 2 else 2,
        'sparse': is_sparse,
        'is_sparse': is_sparse,
        'target_type': 'classification',
        'multilabel': False
    }

    config_space = pipeline.get_configuration_space(dataset_properties, None, None, None, None)


    metalearning_dir = os.path.join(os.path.dirname(metalearning.__file__),
                "files",
                "balanced_accuracy_{0}.classification_{1}".format("multiclass" if dataset_properties["multiclass"] else "binary",
                                                                "sparse" if dataset_properties["sparse"] else "dense"))
    metabase = MetaBase(config_space, metalearning_dir)

    meta_features = None
    try:
        rvals, sparse = perform_one_hot_encoding(dataset_properties["sparse"],
                                    [c in ['categorical'] for c in cat],
                                    [X_train])
        meta_features = _calculate_metafeatures_encoded__(task_id, rvals[0], y_train)
        X_train = rvals
    except:
        meta_features = _calculate_metafeatures__(cat, MULTICLASS_CLASSIFICATION, task_id, X_train, y_train)

    if meta_features is None:
        raise Exception("Error calculating metafeatures")

    metabase.add_dataset(task_id, meta_features)

    configs, list_nn = (suggest_via_metalearning_(metabase, task_id, metric, 2 if dataset_properties["multiclass"] else 1, False, num_initial_configurations))

    return configs, list_nn
