from mosaic_ml.model_config.classification import decision_tree, adaboost, bernouilli_nb, extra_trees,\
    gaussian_nb, gradient_boosting, lda, liblinear_svc, libsvm_svc, multinomial_nb, passive_aggressive,\
    qda, random_forest, sgd, k_nearest_neighbors, xgradient_boosting

def evaluate_classifier(choice, config):
    if choice == "adaboost":
        return adaboost.get_model(choice, config)
    elif choice == "bernoulli_nb":
        return bernouilli_nb.get_model(choice, config)
    elif choice == "decision_tree":
        return decision_tree.get_model(choice, config)
    elif choice == "extra_trees":
        return  extra_trees.get_model(choice, config)
    elif choice == "k_nearest_neighbors":
        return k_nearest_neighbors.get_model(choice, config)
    elif choice == "gaussian_nb":
        return gaussian_nb.get_model(choice, config)
    elif choice == "gradient_boosting":
        return gradient_boosting.get_model(choice, config)
    elif choice == "lda":
        return lda.get_model(choice, config)
    elif choice == "liblinear_svc":
        return liblinear_svc.get_model(choice, config)
    elif choice == "libsvm_svc":
        return libsvm_svc.get_model(choice, config)
    elif choice == "multinomial_nb":
        return multinomial_nb.get_model(choice, config)
    elif choice == "passive_aggressive":
        return  passive_aggressive.get_model(choice, config)
    elif choice == "qda":
        return qda.get_model(choice, config)
    elif choice == "random_forest":
        return random_forest.get_model(choice, config)
    elif choice == "sgd":
        return sgd.get_model(choice, config)
    elif choice == "xgradient_boosting":
        return xgradient_boosting.get_model(choice, config)
    else:
        raise Exception("Classifier {0} not implemented".format(choice))