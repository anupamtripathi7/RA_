"""
Run probit and linear regression for non ranked data
Run for 3 modes (3 capacities) for all cuts and arrivals and save results
"""

import pandas as pd
from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
import statsmodels.api as sm2
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
from stack_unstack import unstack_summary_df
from utils import get_zone_output_path, to_categories
from utils import save_result_to_file
from tqdm import tqdm


zone = '500.0'
cap_mode = 1                # 2 for all
# cut = -1                    # -1 for no choice
# arrival = -1                # -1 for no choice
root = 'data'
results_path = 'results/regression'


def get_regression_features(zone_choice, arrival_choice=-1, cut_choice=-1, cap_mode_choice=2):
    zone_path, zone_file_prefix = get_zone_output_path(zone_choice, root)
    df = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'Summary.csv'))
    df = unstack_summary_df(df, zone=zone_choice).dropna(subset=['capacity'])
    df = df.fillna(0)
    df['cut'] = to_categories(df['cut'])

    if cut_choice in [0, 1]:
        df = df[df['cut'] == cut_choice]
    elif cut_choice == 2:
        df = df[df['cut'] != cut_choice]

    if arrival_choice != -1:
        df = df[df['arrival'] == arrival_choice]

    y = df["discount"]
    if cap_mode_choice == 0:
        x = df[['eco', 'capacity']]
    elif cap_mode_choice == 1:
        x = df[['eco', 'capacity', 'capacity_avg_day']]
    else:
        x = df[['eco', 'capacity', 'capacity_avg_day', 'capacity_avg_global']]
    return x, y, df


def linear_regression_f_score(x, y, arrival_choice, cut_choice, cap_mode_choice=2):
    x = sm.add_constant(x)
    model = sm2.OLS(y, x)
    results = model.fit()
    print(results.summary())
    file_name = 'linear_arrival-{}_cut-{}_cap-{}.txt'.format(arrival_choice, cut_choice, cap_mode_choice)
    with open(os.path.join(os.path.join(results_path, zone), file_name), "w") as text_file:
        text_file.write(str(results.summary()))
    output = {'mode': cap_mode_choice, 'arrival': arrival_choice, 'cut': cut_choice}
    for name, param, std_err in zip(x.columns, results.params, results.bse):
        output[name] = param
        output[name+'_std_err'] = std_err
    output['R2'] = results.rsquared
    output['f_value'] = results.fvalue
    save_result_to_file(pd.DataFrame([output]), 'linear.csv', os.path.join(results_path, zone))
    return results


def probit_parameterized(x, y, arrival_choice=-1, cut_choice=-1, cap_mode_choice=2, save=True):
    """
    Run probit for specified choices
    Args:
        x (pd.DataFrame): Features
        y (pd.DataFrame): Target
        cap_mode_choice (int, optional): eco + 0 for slot capacity only, 1 for slot + daily average, 2 for all
        arrival_choice (int): Choice of arrival day. -1 for all.
        cut_choice: (int): -1 for all, 0 for before cut 1, 2 for before cut 2 and after and 2 for both 0 and 1
        save (bool): If true, saves output in text file

    Returns: Model
    """
    x = sm.add_constant(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = Probit(y_train, x_train)
    probit_model = model.fit()
    print(probit_model.summary())

    r2 = r2_score(y_test, probit_model.predict(x_test))

    if save:
        output = {'mode': cap_mode_choice, 'arrival': arrival_choice, 'cut': cut_choice}
        for key, value in probit_model.params.items():
            output[key] = value
        output['r2'] = r2
        file_name = 'probit_arrival-{}_cut-{}_cap-{}.txt'.format(arrival_choice, cut_choice, cap_mode_choice)
        with open(os.path.join(os.path.join(results_path, zone), file_name), "w") as text_file:
            text_file.write(str(probit_model.summary()))
        save_result_to_file(pd.DataFrame([output]), 'probit.csv', os.path.join(results_path, zone))
    print('R2 score = ', r2)
    return probit_model


def regression_all_cuts_all_arrivals(zone_choice, cap_mode_choice=2):
    arrivals = [-1, 0, 1, 2, 3, 4, 5, 6]
    cuts = [-1, 0, 1, 2]
    for arrival in tqdm(arrivals):
        for cut in cuts:
            features, targets, _ = get_regression_features(zone_choice, cap_mode_choice=cap_mode_choice, arrival_choice=arrival, cut_choice=cut)
            probit_parameterized(features, targets, cap_mode_choice=cap_mode, arrival_choice=arrival, cut_choice=cut)
            linear_regression_f_score(features, targets, arrival, cut, cap_mode_choice=cap_mode)


if __name__ == '__main__':
    regression_all_cuts_all_arrivals(zone, cap_mode)
