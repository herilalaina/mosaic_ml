# Automated machine learning with MCTS
Mosaic ML is a Python library for machine learning pipeline configuration
using Monte Carlo Tree Search.


> **WARNING**: This repository is still under development.



### Installation
```bash
pip install git+https://github.com/herilalaina/mosaic@v0-alpha
pip install git+https://github.com/herilalaina/mosaic_ml
```

### Example of usage: machine learning
A simple example of using `mosaic` to configure machine
learning pipeline.

```bash
cd examples
python simple_example.py
```

If you want to use mosaic_ml inside your code:

```python
from mosaic_ml.automl import AutoML

autoML = AutoML(time_budget=360, scoring_func="balanced_accuracy")
autoML.fit(X_train, y_train)
```
