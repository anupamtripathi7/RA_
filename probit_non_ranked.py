# Do for all cuts and all arrivals
# r2 table for all arrivals
# f value for linear
# Same for ranked
# Add col to steering for each slots with the gr value (from the doc)

"""
Run probit model for non ranked data
"""

import pandas as pd
from statsmodels.discrete.discrete_model import Probit
import statsmodels.tools.tools as sm
from tqdm import tqdm
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split

from utils import get_zone_output_path


zone = '500.0'
cut = 0
arrival = 0
root = 'data'


def unstack_summary_df(summary_df):
    """
    Unstacks the summary dataframe
    Args:
        summary_df (dataframe): summary dataframe

    Returns: unstacked dataframe
    """
    df_info = get_summary_col_info(summary_df)
    slots = df_info['order'][1] - df_info['order'][0]
    slots_per_day = int(slots / 7)
    print(slots)

    # get capacities sums
    cap_start, cap_end = df_info['capacity']
    summary_df['capacity_sum_global'] = summary_df.iloc[:, cap_start: cap_end].sum(axis=1)
    for day in range(7):
        summary_df['capacity_sum_day_' + str(day)] = summary_df.iloc[:, cap_start + (slots_per_day * day): cap_start + (
                    slots_per_day * day) + 9].sum(axis=1)

    df_slots = pd.DataFrame()
    for n in range(slots):
        features = pd.DataFrame()
        features['arrival'] = summary_df['ARRIVAL_DAY']
        features['cut'] = summary_df['ARRIVAL_CAT']
        features['slot'] = n
        features['slot_day'] = n % 7
        for col, (start, end) in df_info.items():
            curr_slot_index = start + n
            features[col] = summary_df.iloc[:, curr_slot_index]
        features['capacity_avg_global'] = (summary_df['capacity_sum_global'] - summary_df.iloc[:, cap_start + n]) / (
                    slots - 1)
        features['capacity_avg_day'] = (summary_df['capacity_sum_day_' + str(n % 7)] - summary_df.iloc[:,
                                                                                       cap_start + n]) / (
                                                   slots_per_day - 1)
        df_slots = df_slots.append(features)
        if n == 25:
            break
    return df_slots


def get_summary_col_info(summary_df, n_slots=9):
    df_info = {}

    col_no = 0
    while col_no < len(summary_df.columns):
        # avail slots
        if summary_df.columns[col_no].startswith('C0_'):
            df_info['avail'] = (col_no, col_no + (7 * n_slots))
            col_no += (7 * n_slots)
        # avail slots
        elif summary_df.columns[col_no].endswith('_Eco'):
            df_info['eco'] = (col_no, col_no + (7 * n_slots))
            col_no += (7 * n_slots)
        # discount slots
        elif summary_df.columns[col_no].endswith('_Discount'):
            df_info['discount'] = (col_no, col_no + (7 * n_slots))
            col_no += (7 * n_slots)
        # capacity slots
        elif summary_df.columns[col_no].endswith('_Capacity'):
            df_info['capacity'] = (col_no, col_no + (7 * n_slots))
            col_no += (7 * n_slots)
        # order slots
        elif summary_df.columns[col_no].startswith('0_'):
            df_info['order'] = (col_no, col_no + (7 * n_slots))
            col_no += (7 * n_slots)
        else:
            col_no += 1
    return df_info


def read_zone_files(zone):
    """
    Reads avail, eco and capacity files for a particular zone
    Args:
        zone (int): Zone to be read
    Returns: Dataframes of the three filesn7 jj8
    """
    avail_df = pd.read_csv(os.path.join(root, str(zone) + '.0_Avail.csv'))
    eco_df = pd.read_csv(os.path.join(root, str(zone) + '.0_Eco.csv'))
    cap_df = pd.read_csv(os.path.join(root, str(zone) + '.0_Cap.csv'))

    return avail_df, eco_df, cap_df


def get_data_for_slots(avail_df, eco_df, cap_df, slots_):   # do for summary
    """
    Returns dataframe with [avail, arrival, cut, slot_day, slot_hour, capacity, discount] columns
    Args:
        avail_df (dataframe): availability csv data
        eco_df (dataframe): eco csv data
        cap_df (dataframe): capacity csv data
        slots_ (list): List of slots
    Returns: Dataframe of required data
    """
    df_slots = pd.DataFrame()
    for n, s in enumerate(tqdm(slots_)):
        features = pd.DataFrame()
        features['avail'] = avail_df[s]
        features['arrival'] = avail_df['ARRIVAL_DAY']
        features['cut'] = avail_df['ARRIVAL_CAT']
        features['slot_day'] = s[1]
        features['slot_hour'] = s[3: 5]
        features['capacity'] = cap_df[s]
        features['discount'] = eco_df[s]
        features['slot'] = s
        df_slots = df_slots.append(features)
        # if n == 25:
        #     break
    return df_slots


def to_categories(data):
    """
    Returns a dataframe with numeric classes starting from 0
    Args:
        data (dataframe): Dataframe to be categorized
    Returns: dataframe with numeric classes
    """
    data = data.astype('category')
    classes = list(range(len(data.unique())))
    data = data.replace(data.unique(), classes)
    return data


def get_avg_capacities(data):
    """
    Returns day and global average for capacities
    Args:
        data (dataframe): Entire dataframe
    Returns: Day and global average for capacities
    """
    return data.groupby(['slot_day']).mean()['capacity'], data['capacity'].mean()


def probit_parameterized(zone_choice, arrival_choice=-1, cut_choice=-1):
    """
    Run probit for specified choices
    Args:
        zone_choice (int): Zone choice
        arrival_choice (int): Arrival day choice
        cut_choice (int): Cut category choice

    Returns: Model
    """
    avail, eco, cap = read_zone_files(zone_choice)

    # Get all slots
    slots = []
    for col in avail.columns[3:-6]:
        slots.append(col)

    df = get_data_for_slots(avail, eco, cap, slots)
    df = df.fillna(0)

    # Consider only slots that are available
    df = df[df['avail'] == 'A']

    # Preprocess df
    df['avail'] = df['avail'] == 'A'
    df['cut'] = to_categories(df['cut'])
    df['slot'] = to_categories(df['slot'])
    df['discount'] = df['discount'].replace('#MULTIVALUE', 0)

    df = df.astype(float)

    df['eco'] = df['discount'].replace([1, 2, 12], [1, 0, 1]).astype(float)
    df['discount'] = df['discount'].replace([1, 2, 12], [0, 1, 1]).astype(float)
    df['capacity'] = df['capacity'].apply(lambda x: max(0, x))

    # Calculate average capacities
    cap_avg_day, cap_avg_global = get_avg_capacities(df)
    df['capacity_avg_global'] = df['capacity'] / cap_avg_global
    df['capacity_avg_day'] = df['capacity']
    for days in df['slot_day'].unique():
        df['capacity_avg_day'][df['slot_day'] == days] = cap_avg_day.loc[days]

    # Slot to one hot
    one_hot = pd.get_dummies(df['slot'])
    df = df.drop('slot', axis=1)
    df = df.join(one_hot)

    if cut_choice != -1:
        df = df[df['cut'] == cut_choice]
    if arrival_choice != -1:
        df = df[df['arrival'] == arrival_choice]

    Y = df["discount"]
    X = df[['eco', 'capacity', 'capacity_avg_day', 'capacity_avg_global']]
    X = sm.add_constant(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    model = Probit(y_train, x_train)
    probit_model = model.fit()
    print(probit_model.summary())

    print(r2_score(y_test, probit_model.predict(x_test)))
    return probit_model
    # print(probit_model.predict(x_test).max())


if __name__ == '__main__':

    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    summary_df = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'Summary.csv'))
