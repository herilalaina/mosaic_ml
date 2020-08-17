# Automated Machine Learning with MCTS

[![Build Status](https://api.travis-ci.org/herilalaina/mosaic_ml.svg?branch=master)](https://travis-ci.org/herilalaina/mosaic_ml)

Mosaic ML is a Python library for machine learning pipeline configuration
using Monte Carlo Tree Search.

The original paper can be found here: [https://www.ijcai.org/Proceedings/2019/457](https://www.ijcai.org/Proceedings/2019/457)

Authors: Herilalaina Rakotoarison, Marc Schoenauer and Michèle Sebag


### Installation

**Requirements**:
* Python (3.5 or higher)
* Numy
* Cython
* scipy
* Mosaic (https://github.com/herilalaina/mosaic)

**Installation**:
```bash
pip install cython numpy scipy pytest
sudo apt-get install build-essential swig
pip install git+https://github.com/herilalaina/mosaic@0.1
pip install git+https://github.com/herilalaina/mosaic_ml
```

### Usage
The entry script is ``python examples/run_mosaic_ml.py -h``.

```
--openml-task-id OPENML_TASK_ID
                      OpenML Task ID (default 252)
--overall-time-budget OVERALL_TIME_BUDGET
                      Overall time budget in seconds (default 360)
--eval-time-budget EVAL_TIME_BUDGET
                      Time budget for each machine learning evaluation
                      (default 100)
--memory-limit MEMORY_LIMIT
                      RAM Memory limit (default 3034)
--seed SEED           Seed for reproducibility (default 42)
--nb-init-metalearning NB_INIT_METALEARNING
                      Number of initial configurations from Auto-Sklearn
                      (default 25)
--ensemble-size ENSEMBLE_SIZE
                      Size of ensemble set (default 50)
```


**Mosaic ML** has three different components:
* *vanilla*: MCTS for algorithm selection and Bayesian Optimization for hyperparameter tuning

```bash
python examples/run_mosaic_ml.py --nb-init-metalearning 0 --ensemble-size 1
```

* *metalearning*: initialize with a set of configurations fetched from [Auto-Sklearn](https://automl.github.io/auto-sklearn/master/index.html) then apply *vanilla setting*

```bash
python examples/run_mosaic_ml.py --nb-init-metalearning 25 --ensemble-size 1
```

* *ensemble (with metalearning)*: add an ensemble selection method ([Caruana et al, 04](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)) in the top of the *metalearning* setting


```bash
python examples/run_mosaic_ml.py --nb-init-metalearning 25 --ensemble-size 50
```
