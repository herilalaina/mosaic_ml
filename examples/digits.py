import sys
sys.path.insert(0,'/home/herilalaina/Code/software/mosaic_ml')

import unittest
import warnings

from sklearn import datasets, feature_selection, linear_model, feature_selection
from sklearn.model_selection import train_test_split

from mosaic_ml.automl import AutoML

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

autoML = AutoML()
autoML.fit(X_train, y_train)
