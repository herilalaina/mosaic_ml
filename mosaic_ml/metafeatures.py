import openml
import numpy as np

list_metafeatures = ['MaxMutualInformation',
 'Quartile3MutualInformation',
 'NumberOfInstances',
 'NumberOfNumericFeatures',
 'MinSkewnessOfNumericAtts',
 'ClassEntropy',
 'MinMutualInformation',
 'MaxSkewnessOfNumericAtts',
 'MaxKurtosisOfNumericAtts',
 'PercentageOfSymbolicFeatures',
 'MeanSkewnessOfNumericAtts',
 'EquivalentNumberOfAtts',
 'MajorityClassPercentage',
 'NumberOfMissingValues',
 'Quartile2KurtosisOfNumericAtts',
 'Quartile1KurtosisOfNumericAtts',
 'NumberOfFeatures',
 'NumberOfInstancesWithMissingValues',
 'Quartile3StdDevOfNumericAtts',
 'Quartile1StdDevOfNumericAtts',
 'PercentageOfBinaryFeatures',
 'MeanAttributeEntropy',
 'MinorityClassSize',
 'PercentageOfInstancesWithMissingValues',
 'MaxNominalAttDistinctValues',
 'MinorityClassPercentage',
 'StdvNominalAttDistinctValues',
 'MaxMeansOfNumericAtts',
 'MeanStdDevOfNumericAtts',
 'Quartile2StdDevOfNumericAtts',
 'Quartile2AttributeEntropy',
 'Quartile3SkewnessOfNumericAtts',
 'Quartile1AttributeEntropy',
 'Quartile1MeansOfNumericAtts',
 'PercentageOfNumericFeatures',
 'MinKurtosisOfNumericAtts',
 'PercentageOfMissingValues',
 'MinMeansOfNumericAtts',
 'NumberOfBinaryFeatures',
 'MinAttributeEntropy',
 'MeanMeansOfNumericAtts',
 'Quartile3AttributeEntropy',
 'NumberOfClasses',
 'MajorityClassSize',
 'MeanKurtosisOfNumericAtts',
 'Quartile2MutualInformation',
 'Quartile3KurtosisOfNumericAtts',
 'Quartile1SkewnessOfNumericAtts',
 'Quartile3MeansOfNumericAtts',
 'Quartile2SkewnessOfNumericAtts',
 'AutoCorrelation',
 'MinStdDevOfNumericAtts',
 'MeanMutualInformation',
 'MeanNominalAttDistinctValues',
 'Quartile1MutualInformation',
 'NumberOfSymbolicFeatures',
 'MaxAttributeEntropy',
 'Quartile2MeansOfNumericAtts',
 'MaxStdDevOfNumericAtts',
 'Dimensionality',
 'MinNominalAttDistinctValues']

def get_dataset_metafeature_from_openml(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)
    features = []
    for f in list_metafeatures:
        try:
            val = dataset.qualities[f]
            if not np.nan(val):
                features.append(val)
            else:
                features.append(0)
        except:
            features.append(0)
    return features
