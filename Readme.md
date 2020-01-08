[![CircleCI](https://circleci.com/gh/herilalaina/mosaic_ml/tree/master.svg?style=svg)](https://circleci.com/gh/herilalaina/mosaic_ml/tree/master)


# Automated machine learning with MCTS

Mosaic ML is a Python library for machine learning pipeline configuration
using Monte Carlo Tree Search.


> **WARNING**: This repository is still under development.



### Installation
`Mosaic ML` relies on the pipeline optimization library [`mosaic`](https://github.com/herilalaina/mosaic).
 

```bash
pip install git+https://github.com/herilalaina/mosaic@v0-alpha
pip install git+https://github.com/herilalaina/mosaic_ml
```

### Usage
A [simple example](https://github.com/herilalaina/mosaic_ml/blob/master/examples/simple_example.py) of using `mosaic` to configure machine learning pipeline.


```python
from mosaic_ml.automl import 

X_train, y_train, X_test, y_test, cat = load_task(6)

autoML = AutoML(time_budget=120,
                time_limit_for_evaluation=100,
                memory_limit=3024,
                seed=1,
                scoring_func="balanced_accuracy",
                exec_dir="execution_dir"
                )

best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
print(autoML.get_run_history())
```

### Citation
```
@inproceedings{ijcai2019-457,
  title     = {Automated Machine Learning with Monte-Carlo Tree Search},
  author    = {Rakotoarison, Herilalaina and Schoenauer, Marc and Sebag, Michèle},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {3296--3303},
  year      = {2019},
  month     = {7},
  doi       = {10.24963/ijcai.2019/457},
  url       = {https://doi.org/10.24963/ijcai.2019/457},
}
```
