from mosaic_ml.automl import AutoML
from update_metadata_util import load_task

if __name__=="__main__":
    X_train, y_train, X_test, y_test, cat = load_task(252)
    autoML = AutoML(time_budget=360,
                    time_limit_for_evaluation=100,
                    memory_limit=3024,
                    seed=1,
                    scoring_func="balanced_accuracy",
                    verbose=True,
                    ensemble_size=0
                    )

    best_config, best_score = autoML.fit(X_train, y_train, X_test, y_test, categorical_features=cat)
    run = autoML.get_run_history()[-1] # Get last improvement
    print("Best config {0}\t Validation Score:{1}\t Test Score:{2}".format(run["model"], run["validation_score"], run["test_score"]))
