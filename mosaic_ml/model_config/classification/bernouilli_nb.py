from autosklearn.pipeline.components.classification.bernoulli_nb import BernoulliNB


def get_model(name, config, random_state):
    model = BernoulliNB(
        alpha=config["classifier:bernoulli_nb:alpha"],
        fit_prior=config["classifier:bernoulli_nb:fit_prior"],
        random_state=random_state
    )
    return (name, model)
