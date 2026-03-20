import numpy as np

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))

def run_ts_cross_validation(df, model, horizon, n_folds=3, step_size=24):
    errors = []
    for i in range(n_folds):
        cutoff = len(df) - (i * step_size)
        if cutoff <= horizon + 48:
            break

        train = df.iloc[:cutoff - horizon]
        test = df.iloc[cutoff - horizon : cutoff]['y'].values

        model.fit(train, horizon)
        forecast = model.predict(train, horizon)
        errors.append(smape(test, forecast))
    return errors
