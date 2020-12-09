"""
Run probit model for non ranked data
Run for 3 modes (3 capacities) and save results
"""

import pandas as pd
from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
from stack_unstack import unstack_summary_df
from utils import get_zone_output_path, to_categories
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from utils import save_result_to_file


zone = '500.0'
cut = 0
arrival = 0
root = 'data'
results_path = 'results/regression'


def get_regression_features(zone, arrival_choice=-1, cut_choice=-1, cap_mode=0):
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    df = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'Summary.csv'))
    df = unstack_summary_df(df).dropna(subset=['capacity'])
    df = df.fillna(0)
    df['cut'] = to_categories(df['cut'])

    if cut_choice != -1:
        df = df[df['cut'] == cut_choice]
    if arrival_choice != -1:
        df = df[df['arrival'] == arrival_choice]

    y = df["discount"]
    if cap_mode == 0:
        x = df[['eco', 'capacity']]
    elif cap_mode == 1:
        x = df[['eco', 'capacity', 'capacity_avg_day']]
    else:
        x = df[['eco', 'capacity', 'capacity_avg_day', 'capacity_avg_global']]
    return x, y


def linear_regression_f_score(x, y):
    feature_names = ['constant'] + x.columns.to_list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    reg = LinearRegression().fit(x_train, y_train)
    r2 = reg.score(x_test, y_test)
    return f_regression(x, y)


def probit_parameterized(x, y, cap_mode=2):
    """
    Run probit for specified choices
    Args:
        x (pd.DataFrame): Features
        y (pd.DataFrame): Target
        cap_mode (int, optional): 0 for slot capacity only, 1 for slot + week average, 2 for all

    Returns: Model
    """
    x = sm.add_constant(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    model = Probit(y_train, x_train)
    probit_model = model.fit()
    print(probit_model.summary())

    r2 = r2_score(y_test, probit_model.predict(x_test))

    output = {'mode': cap_mode}
    for key, value in probit_model.params.items():
        output[key] = value
    output['r2'] = r2
    save_result_to_file(pd.DataFrame([output]), 'probit.csv', results_path)
    print('R2 score = ', r2)


if __name__ == '__main__':
    features, targets = get_regression_features(zone, cap_mode=1)
    probit_parameterized(features, targets, cap_mode=1)
    # print(linear_regression_f_score(features, targets))
