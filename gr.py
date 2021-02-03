"""
Fill in gr values for all slots for non ranked data
"""

import pandas as pd
import statsmodels.tools.tools as sm
import os
from scipy.stats import norm
from stack_unstack import stack_df
from regression_non_ranked import get_regression_features, probit_parameterized
from regression_ranked import get_regression_features_ranked, probit_parameterized_ranked
from tqdm import tqdm


zone = '500.0'
root = 'data'
result_path = 'results/gr'


def fill_gr(y, y_pred):
    """
    Get gr values
    Args:
        y: Targets
        y_pred: Predicted values
    Returns: gr values
    """
    imr = norm.pdf(y_pred) / norm.cdf(y_pred)
    return y * imr - (1 - y) * imr


def gr_non_ranked(zone_choice, cap_mode_choice=2):
    """
        Calculate gr values for each slot of the ranked data
        Args:
            zone_choice (str): Zone number in float string
            cap_mode_choice (int, optional): 0 for eco + slot capacity only, 1 for slot + daily average, 2 for all
        Returns:
            (pd.DataFrame): Stacked version of summary file with gr values
        """
    _, _, df = get_regression_features(zone_choice)
    y = df["discount"]
    if cap_mode_choice == 0:
        x = df[['capacity', 'eco']]
    elif cap_mode_choice == 1:
        x = df[['capacity', 'capacity_avg_day', 'eco']]
    else:
        x = df[['capacity', 'capacity_avg_global', 'capacity_avg_day', 'eco']]
    x = sm.add_constant(x)
    model = probit_parameterized(x, y, save=False)

    gr = fill_gr(y, model.predict(x))
    df = pd.concat([df, gr.reindex(df.index)], axis=1)
    df = stack_df(df, save=False)
    print(df, df.columns)
    df.to_csv(os.path.join(result_path, 'gr_unranked_stacked.csv'))


def gr_ranked(zone_choice, cap_mode_choice=2):
    """
    Calculate gr values for each slot of the ranked data
    Args:
        zone_choice (str): Zone number in float string
        cap_mode_choice (int, optional): 0 for eco + slot capacity only, 1 for slot + daily average, 2 for all
    Returns:
        (pd.DataFrame): Stacked version of summary file with gr values
    """
    _, _, df = get_regression_features_ranked(zone_choice)
    y = df["discount"]
    x = df.drop(['discount', 'day', 'order', 'avail', 'primary_key', 'slot'], axis=1)
    if cap_mode_choice == 0:
        x = df.drop(['capacity_avg_global', 'capacity_avg_day'])
    elif cap_mode_choice == 1:
        x = df.drop(['capacity_avg_day'])
    x = sm.add_constant(x)
    model = probit_parameterized_ranked(x, y, save=False)

    gr = fill_gr(y, model.predict(x))
    df = pd.concat([df, gr.reindex(df.index)], axis=1)
    df = stack_df(df, save=False)
    df.to_csv(os.path.join(result_path, 'gr_ranked_stacked.csv'))
    return df


def gr_non_ranked_all_cuts_all_arrivals(zone_choice, cap_mode_choice):
    """
    Calculate gr values for each slots for all arrivals and cuts.
    Args:
        zone_choice (str): Zone number in float string
        cap_mode_choice (int, optional): 0 for eco + slot capacity only, 1 for slot + daily average, 2 for all
    """
    arrivals = [-1, 0, 1, 2, 3, 4, 5, 6]
    cuts = [-1, 0, 1, 2]
    for arrival in tqdm(arrivals):
        for cut in cuts:
            gr_non_ranked(zone_choice)

if __name__ == "__main__":
    gr_ranked(zone)
