from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

def get_model(name, config):
    score_name = config["preprocessor:select_percentile_classification:score_func"]
    if score_name == "chi2":
        score = chi2
    elif score_name == "f_classif":
        score = f_classif
    elif score_name == "mutual_info":
        score = mutual_info_classif

    model = SelectPercentile(
        percentile=float(config["preprocessor:select_percentile_classification:percentile"]),
        score_func=score
    )
    return (name, model)