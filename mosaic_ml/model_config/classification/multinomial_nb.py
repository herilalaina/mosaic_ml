from sklearn.naive_bayes import MultinomialNB


def get_model(name, config):
    model = MultinomialNB(
        alpha=float(config["classifier:multinomial_nb:alpha"]),
        fit_prior=eval(config["classifier:multinomial_nb:fit_prior"])
    )
    return (name, model)
