from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif, chi2


def get_model(name, config):
    alpha = float(config["preprocessor:select_rates:alpha"])
    name = config["preprocessor:select_rates:mode"]
    score_name = config["preprocessor:select_rates:score_func"]
    if score_name == "chi2":
        score = chi2
    elif score_name == "f_classif":
        score = f_classif

    if name == "fpr":
        return (name, SelectFpr(score_func=score, alpha=alpha))
    elif name == "fdr":
        return (name, SelectFdr(score_func=score, alpha=alpha))
    else:
        return (name, SelectFwe(score_func=score, alpha=alpha))