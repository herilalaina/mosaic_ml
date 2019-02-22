from xgboost import XGBClassifier


def get_model(name, config, random_state):
    model = XGBClassifier(
        base_score=float(config["classifier:xgradient_boosting:base_score"]),
        booster=config["classifier:xgradient_boosting:booster"],
        colsample_bylevel=float(config["classifier:xgradient_boosting:colsample_bylevel"]),
        colsample_bytree=float(config["classifier:xgradient_boosting:colsample_bytree"]),
        gamma=float(config["classifier:xgradient_boosting:gamma"]),
        learning_rate=float(config["classifier:xgradient_boosting:learning_rate"]),
        max_delta_step=int(config["classifier:xgradient_boosting:max_delta_step"]),
        max_depth=int(config["classifier:xgradient_boosting:max_depth"]),
        min_child_weight=int(config["classifier:xgradient_boosting:min_child_weight"]),
        n_estimators=int(config["classifier:xgradient_boosting:n_estimators"]),
        reg_alpha=float(config["classifier:xgradient_boosting:reg_alpha"]),
        reg_lambda=float(config["classifier:xgradient_boosting:reg_lambda"]),
        scale_pos_weight=float(config["classifier:xgradient_boosting:scale_pos_weight"]),
        subsample=float(config["classifier:xgradient_boosting:subsample"]),
        n_jobs=1,
        random_state=random_state
    )
    return (name, model)
