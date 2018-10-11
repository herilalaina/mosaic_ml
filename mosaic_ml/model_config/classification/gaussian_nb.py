from sklearn.naive_bayes import GaussianNB

def get_model(name, config):
    return (name, GaussianNB())