from autosklearn.pipeline.components.classification.gaussian_nb import GaussianNB


def get_model(name, config, random_state):
    return (name, GaussianNB(random_state=random_state))
