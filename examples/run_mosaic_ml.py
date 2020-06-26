import argparse
from mosaic_ml.automl import AutoML
from update_metadata_util import load_task
from mosaic_ml.autosklearn_wrapper.autosklearn import get_autosklearn_metalearning

parser = argparse.ArgumentParser(description='Mosaic for AutoML')

parser.add_argument('--openml-task-id', type=int, default=252,
                    help='OpenML Task ID (default 252)')
parser.add_argument('--overall-time-budget', type=int, default=360,
                    help='Overall time budget in seconds (default 360)')
parser.add_argument('--eval-time-budget', type=int, default=360,
                    help='Time budget for each machine learning evaluation (default 100)')
parser.add_argument('--memory-limit', type=int, default=3024,
                    help='RAM Memory limit (default 3034)')
parser.add_argument('--seed', type=int, default=42,
                    help='Seed for reproducibility (default 42)')
parser.add_argument('--nb-init-metalearning', type=int, default=25,
                    help='Number of initial configurations from Auto-Sklearn (default 25)')
parser.add_argument('--ensemble-size', type=int, default=50,
                    help='Size of ensemble set (default 50)')


if __name__=="__main__":
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, cat = load_task(252)
    autoML = AutoML(time_budget=args.overall_time_budget,
                    time_limit_for_evaluation=args.eval_time_budget,
                    memory_limit=args.memory_limit,
                    seed=args.seed,
                    scoring_func="balanced_accuracy",
                    ensemble_size=args.ensemble_size,
                    verbose=True
                    )

    if args.nb_init_metalearning > 0:
        intial_configuration_metalearning_AS, _ = get_autosklearn_metalearning(X_train, y_train, cat, "balanced_accuracy", args.nb_init_metalearning) # init with 25 configs
    else:
        intial_configuration_metalearning_AS = []
    best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test,
                                            categorical_features=cat,
                                            initial_configurations=intial_configuration_metalearning_AS)
    run = autoML.get_run_history()[-1] # Get last improvement
    print("Best config {0}\t Validation Score:{1}\t Test Score:{2}".format(run["model"], run["validation_score"], run["test_score"]))
