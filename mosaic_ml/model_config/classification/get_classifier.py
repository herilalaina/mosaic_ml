from mosaic_ml.model_config.classification import decision_tree, adaboost, bernouilli_nb, extra_trees, \
    gaussian_nb, gradient_boosting, lda, liblinear_svc, libsvm_svc, multinomial_nb, passive_aggressive, \
    qda, random_forest, sgd, k_nearest_neighbors, xgradient_boosting, logistc_regression


def evaluate_classifier(choice, config, random_state):
    if choice == "adaboost":
        return adaboost.get_model(choice, config, random_state)
    elif choice == "bernoulli_nb":
        return bernouilli_nb.get_model(choice, config, random_state)
    elif choice == "decision_tree":
        return decision_tree.get_model(choice, config, random_state)
    elif choice == "extra_trees":
        return extra_trees.get_model(choice, config, random_state)
    elif choice == "k_nearest_neighbors":
        return k_nearest_neighbors.get_model(choice, config, random_state)
    elif choice == "gaussian_nb":
        return gaussian_nb.get_model(choice, config, random_state)
    elif choice == "gradient_boosting":
        return gradient_boosting.get_model(choice, config, random_state)
    elif choice == "lda":
        return lda.get_model(choice, config, random_state)
    elif choice == "liblinear_svc":
        return liblinear_svc.get_model(choice, config, random_state)
    elif choice == "libsvm_svc":
        return libsvm_svc.get_model(choice, config, random_state)
    elif choice == "logistic_regression":
        return logistc_regression.get_model(choice, config, random_state)
    elif choice == "multinomial_nb":
        return multinomial_nb.get_model(choice, config, random_state)
    elif choice == "passive_aggressive":
        return passive_aggressive.get_model(choice, config, random_state)
    elif choice == "qda":
        return qda.get_model(choice, config, random_state)
    elif choice == "random_forest":
        return random_forest.get_model(choice, config, random_state)
    elif choice == "sgd":
        return sgd.get_model(choice, config, random_state)
    elif choice == "xgradient_boosting":
        return xgradient_boosting.get_model(choice, config, random_state)
    else:
        raise Exception("Classifier {0} not implemented".format(choice))
