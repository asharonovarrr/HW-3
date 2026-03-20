import pandas as pd
import numpy as np
import os
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

import config
from data_loader import load_m4_data
from models import StatsForecastWrapper, DirectCatBoostModel, StackingEnsemble
from statsforecast.models import AutoTheta
from validation import run_ts_cross_validation

def process_single_series(i, series, models_dict):
    series_results = []
    for model_name, model in models_dict.items():
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'set_params'):
                model.model.set_params(thread_count=1)

            errors = run_ts_cross_validation(
                series, model, config.HORIZON,
                n_folds=config.N_FOLDS, step_size=config.STEP_SIZE
            )

            if errors:
                series_results.append({
                    "series_id": i,
                    "model": model_name,
                    "mean_smape": np.mean(errors),
                    "std_smape": np.std(errors)
                })
        except:
            pass
    return series_results

def main():
    data = load_m4_data(n_series=config.N_SERIES)

    theta_model = StatsForecastWrapper(AutoTheta())
    catboost_direct = DirectCatBoostModel(window=config.WINDOW_SIZE, iterations=config.ITERATIONS)
    ensemble = StackingEnsemble([theta_model, catboost_direct])

    models_dict = {
        "AutoTheta (Local Stat)": theta_model,
        "CatBoost (Direct ML)": catboost_direct,
        "Stacking Ensemble": ensemble
    }

    print(f"Running: {config.N_SERIES} series")

    nested = Parallel(n_jobs=-1)(
        delayed(process_single_series)(i, series, models_dict)
        for i, series in enumerate(tqdm(data))
    )

    results = [item for sublist in nested for item in sublist]
    df_results = pd.DataFrame(results)

    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/metrics.csv", index=False)

    summary = df_results.groupby("model")[["mean_smape", "std_smape"]].mean().sort_values("mean_smape")
    print("\nResults:")
    print(summary)

if __name__ == "__main__":
    main()
