"""
Fill in gr values for all slots for non ranked data
"""

import pandas as pd
import statsmodels.tools.tools as sm
import os
from scipy.stats import norm
from stack_unstack import stack_df, unstack_summary_df
from regression_non_ranked import get_regression_features, probit_parameterized


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


def gr_non_ranked(zone_choice):
    _, _, df = get_regression_features(zone_choice)
    y = df["discount"]
    x = df.drop(['discount', 'day', 'order', 'avail', 'primary_key'], axis=1)
    x = sm.add_constant(x)
    model = probit_parameterized(x, y)

    gr = fill_gr(y, model.predict(x))
    df = pd.concat([df, gr.reindex(df.index)], axis=1)
    df = stack_df(df, save=False)
    print(df, df.columns)
    df.to_csv(os.path.join(result_path, 'gr_unranked_stacked.csv'))


if __name__ == "__main__":
    gr_non_ranked(zone)

