# Automated machine learning with MCTS
Mosaic ML is a Python library for machine learning pipeline configuration 
using Monte Carlo Tree Search.

### Example of usage: machine learning
A simple example of using `mosaic` to configure machine 
learning pipeline.

```python
from mosaic_ml.automl import AutoML

autoML = AutoML(time_budget=360, scoring_func="balanced_accuracy")
autoML.fit(X_train, y_train)
```
