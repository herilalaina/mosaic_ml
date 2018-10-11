from sklearn.naive_bayes import BernoulliNB


def get_model(name, config):
    model = BernoulliNB(
        alpha=float(config["classifier:bernoulli_nb:alpha"]),
        fit_prior=eval(config["classifier:bernoulli_nb:fit_prior"])
    )
    return (name, model)
