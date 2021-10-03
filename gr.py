"""
Fill in gr values for all slots for non ranked data
remove 1 slot info from x line 56...
1-inv mills ratio
"""

import pandas as pd
import statsmodels.tools.tools as sm
import os
from scipy.stats import norm
from stack_unstack import stack_df, stack_df_ranked
from regression_non_ranked import get_regression_features, probit_parameterized
from regression_ranked import get_regression_features_ranked, probit_parameterized_ranked
from tqdm import tqdm
import numpy as np


zone = '500.0'
root = 'data'
result_path = 'results/gr'


def inverse_mills_ratio(y_pred):
    """
    Return the inverse mills ratio for the given input y pred.
    Args:
        y_pred
    """
    return norm.pdf(y_pred) / norm.cdf(y_pred)          # changed to 1-cdf  # 7/12/2020 check without 1 - cdf


def fill_gr(y, y_pred):
    """
    Get gr values
    Args:
        y: Targets
        y_pred: Predicted values
    Returns: gr values
    """
    return y * inverse_mills_ratio(y_pred) - (1 - y) * inverse_mills_ratio(-1 * y_pred)


def gr_non_ranked(zone_choice, cap_mode_choice=2, arrival_choice=0, cut_choice=0):
    """
        Calculate gr values for each slot of the ranked data
        Args:
            zone_choice (str): Zone number in float string
            cap_mode_choice (int, optional): 0 for eco + slot capacity only, 1 for slot + daily average, 2 for all
            arrival_choice
            cut_choice
        Returns:
            (pd.DataFrame): Stacked version of summary file with gr values
        """
    _, _, df = get_regression_features(zone_choice, arrival_choice=arrival_choice, cut_choice=cut_choice)
    y = df["discount"]
    if cap_mode_choice == 0:
        x = df[['capacity', 'eco'] + ['slot_0_06:30 - 08:00', 'slot_0_08:00 - 10:00', 'slot_0_10:00 - 12:00', 'slot_0_12:00 - 14:00', 'slot_0_14:00 - 16:00', 'slot_0_16:00 - 18:00', 'slot_0_18:00 - 20:00', 'slot_0_20:00 - 22:00', 'slot_0_22:00 - 23:30', 'slot_1_06:30 - 08:00', 'slot_1_08:00 - 10:00', 'slot_1_10:00 - 12:00', 'slot_1_12:00 - 14:00', 'slot_1_14:00 - 16:00', 'slot_1_16:00 - 18:00', 'slot_1_18:00 - 20:00', 'slot_1_20:00 - 22:00', 'slot_1_22:00 - 23:30', 'slot_2_06:30 - 08:00', 'slot_2_08:00 - 10:00', 'slot_2_10:00 - 12:00', 'slot_2_12:00 - 14:00', 'slot_2_14:00 - 16:00', 'slot_2_16:00 - 18:00', 'slot_2_18:00 - 20:00', 'slot_2_20:00 - 22:00', 'slot_2_22:00 - 23:30', 'slot_3_06:30 - 08:00', 'slot_3_08:00 - 10:00', 'slot_3_10:00 - 12:00', 'slot_3_12:00 - 14:00', 'slot_3_14:00 - 16:00', 'slot_3_16:00 - 18:00', 'slot_3_18:00 - 20:00', 'slot_3_20:00 - 22:00', 'slot_3_22:00 - 23:30', 'slot_4_06:30 - 08:00', 'slot_4_08:00 - 10:00', 'slot_4_10:00 - 12:00', 'slot_4_12:00 - 14:00', 'slot_4_14:00 - 16:00', 'slot_4_16:00 - 18:00', 'slot_4_18:00 - 20:00', 'slot_4_20:00 - 22:00', 'slot_4_22:00 - 23:30', 'slot_5_06:30 - 08:00', 'slot_5_08:00 - 10:00', 'slot_5_10:00 - 12:00', 'slot_5_12:00 - 14:00', 'slot_5_14:00 - 16:00', 'slot_5_16:00 - 18:00', 'slot_5_18:00 - 20:00', 'slot_5_20:00 - 22:00', 'slot_5_22:00 - 23:30', 'slot_6_06:30 - 08:00', 'slot_6_08:00 - 10:00', 'slot_6_10:00 - 12:00', 'slot_6_12:00 - 14:00', 'slot_6_14:00 - 16:00', 'slot_6_16:00 - 18:00', 'slot_6_18:00 - 20:00', 'slot_6_20:00 - 22:00', 'slot_6_22:00 - 23:30', 'slot_NO_PURCHASE']]
    elif cap_mode_choice == 1:
        x = df[['capacity', 'capacity_avg_day', 'eco'] + ['slot_0_06:30 - 08:00', 'slot_0_08:00 - 10:00', 'slot_0_10:00 - 12:00', 'slot_0_12:00 - 14:00', 'slot_0_14:00 - 16:00', 'slot_0_16:00 - 18:00', 'slot_0_18:00 - 20:00', 'slot_0_20:00 - 22:00', 'slot_0_22:00 - 23:30', 'slot_1_06:30 - 08:00', 'slot_1_08:00 - 10:00', 'slot_1_10:00 - 12:00', 'slot_1_12:00 - 14:00', 'slot_1_14:00 - 16:00', 'slot_1_16:00 - 18:00', 'slot_1_18:00 - 20:00', 'slot_1_20:00 - 22:00', 'slot_1_22:00 - 23:30', 'slot_2_06:30 - 08:00', 'slot_2_08:00 - 10:00', 'slot_2_10:00 - 12:00', 'slot_2_12:00 - 14:00', 'slot_2_14:00 - 16:00', 'slot_2_16:00 - 18:00', 'slot_2_18:00 - 20:00', 'slot_2_20:00 - 22:00', 'slot_2_22:00 - 23:30', 'slot_3_06:30 - 08:00', 'slot_3_08:00 - 10:00', 'slot_3_10:00 - 12:00', 'slot_3_12:00 - 14:00', 'slot_3_14:00 - 16:00', 'slot_3_16:00 - 18:00', 'slot_3_18:00 - 20:00', 'slot_3_20:00 - 22:00', 'slot_3_22:00 - 23:30', 'slot_4_06:30 - 08:00', 'slot_4_08:00 - 10:00', 'slot_4_10:00 - 12:00', 'slot_4_12:00 - 14:00', 'slot_4_14:00 - 16:00', 'slot_4_16:00 - 18:00', 'slot_4_18:00 - 20:00', 'slot_4_20:00 - 22:00', 'slot_4_22:00 - 23:30', 'slot_5_06:30 - 08:00', 'slot_5_08:00 - 10:00', 'slot_5_10:00 - 12:00', 'slot_5_12:00 - 14:00', 'slot_5_14:00 - 16:00', 'slot_5_16:00 - 18:00', 'slot_5_18:00 - 20:00', 'slot_5_20:00 - 22:00', 'slot_5_22:00 - 23:30', 'slot_6_06:30 - 08:00', 'slot_6_08:00 - 10:00', 'slot_6_10:00 - 12:00', 'slot_6_12:00 - 14:00', 'slot_6_14:00 - 16:00', 'slot_6_16:00 - 18:00', 'slot_6_18:00 - 20:00', 'slot_6_20:00 - 22:00', 'slot_6_22:00 - 23:30', 'slot_NO_PURCHASE']]
    else:
        x = df[['capacity', 'capacity_avg_global', 'capacity_avg_day', 'eco'] + ['slot_0_06:30 - 08:00', 'slot_0_08:00 - 10:00', 'slot_0_10:00 - 12:00', 'slot_0_12:00 - 14:00', 'slot_0_14:00 - 16:00', 'slot_0_16:00 - 18:00', 'slot_0_18:00 - 20:00', 'slot_0_20:00 - 22:00', 'slot_0_22:00 - 23:30', 'slot_1_06:30 - 08:00', 'slot_1_08:00 - 10:00', 'slot_1_10:00 - 12:00', 'slot_1_12:00 - 14:00', 'slot_1_14:00 - 16:00', 'slot_1_16:00 - 18:00', 'slot_1_18:00 - 20:00', 'slot_1_20:00 - 22:00', 'slot_1_22:00 - 23:30', 'slot_2_06:30 - 08:00', 'slot_2_08:00 - 10:00', 'slot_2_10:00 - 12:00', 'slot_2_12:00 - 14:00', 'slot_2_14:00 - 16:00', 'slot_2_16:00 - 18:00', 'slot_2_18:00 - 20:00', 'slot_2_20:00 - 22:00', 'slot_2_22:00 - 23:30', 'slot_3_06:30 - 08:00', 'slot_3_08:00 - 10:00', 'slot_3_10:00 - 12:00', 'slot_3_12:00 - 14:00', 'slot_3_14:00 - 16:00', 'slot_3_16:00 - 18:00', 'slot_3_18:00 - 20:00', 'slot_3_20:00 - 22:00', 'slot_3_22:00 - 23:30', 'slot_4_06:30 - 08:00', 'slot_4_08:00 - 10:00', 'slot_4_10:00 - 12:00', 'slot_4_12:00 - 14:00', 'slot_4_14:00 - 16:00', 'slot_4_16:00 - 18:00', 'slot_4_18:00 - 20:00', 'slot_4_20:00 - 22:00', 'slot_4_22:00 - 23:30', 'slot_5_06:30 - 08:00', 'slot_5_08:00 - 10:00', 'slot_5_10:00 - 12:00', 'slot_5_12:00 - 14:00', 'slot_5_14:00 - 16:00', 'slot_5_16:00 - 18:00', 'slot_5_18:00 - 20:00', 'slot_5_20:00 - 22:00', 'slot_5_22:00 - 23:30', 'slot_6_06:30 - 08:00', 'slot_6_08:00 - 10:00', 'slot_6_10:00 - 12:00', 'slot_6_12:00 - 14:00', 'slot_6_14:00 - 16:00', 'slot_6_16:00 - 18:00', 'slot_6_18:00 - 20:00', 'slot_6_20:00 - 22:00', 'slot_6_22:00 - 23:30', 'slot_NO_PURCHASE']]
    x = sm.add_constant(x)
    model = probit_parameterized(x, y, save=False)

    gr = fill_gr(y, model.predict(x, linear=True))      # checked
    df = pd.concat([df, gr.reindex(df.index).rename('gr')], axis=1)
    df = stack_df(df, save=False, gr=True)
    df.to_csv(os.path.join(result_path, zone_choice, 'gr_unranked_stacked_arrival_{}_cut_{}_cap_{}.csv'.format(arrival_choice, cut_choice, cap_mode_choice)))


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
    model = probit_parameterized_ranked(x, y, -1, -1, save=False)

    gr = fill_gr(y, model.predict(x))
    df = pd.concat([df, gr.reindex(df.index)], axis=1)
    df = stack_df_ranked(df, gr=True)
    df.to_csv(os.path.join(result_path, 'gr_ranked_stacked_cap_{}.csv'.format(cap_mode_choice)))
    return df


def gr_non_ranked_all_cuts_all_arrivals(zone_choice, cap_mode_choice=2):
    """
    Calculate gr values for each slots for all arrivals and cuts.
    Args:
        zone_choice (str): Zone number in float string
        cap_mode_choice (int, optional): 0 for eco + slot capacity only, 1 for slot + daily average, 2 for all
    """
    arrivals = [0]
    cuts = [0]
    for arrival in tqdm(arrivals):
        for cut in cuts:
            gr_non_ranked(zone_choice, cap_mode_choice=cap_mode_choice, arrival_choice=arrival, cut_choice=cut)


if __name__ == "__main__":
    gr_non_ranked_all_cuts_all_arrivals(zone, cap_mode_choice=2)
