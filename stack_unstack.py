# ToDO: added no purchase to unstack non ranked. Test remaining


import pandas as pd
from tqdm import tqdm
import os
import utils as u
from utils import get_zone_output_path


zone = '500.0'
root = 'data'
results_path = 'results'


def unstack_summary_df(summary_df, zone, root='data', check_saved=False, save=False):  # do for summary
    """
    Unstacks the summary dataframe
    Args:
        summary_df (dataframe): summary dataframe

    Returns: unstacked dataframe
    """
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    if os.path.exists(os.path.join(zone_path, zone_file_prefix + 'unstacked.csv')) and check_saved:
        return pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'unstacked.csv'), index_col=0)
    summary_df['primary_key'] = summary_df['EVENT_DTM'].astype(str) + '-' + summary_df['CUSTOMER_ID'].astype(str)
    slots_per_day = u.get_slots_per_day_for_zone(zone, root)
    slots_offered = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'SlotsOfferedTitle.csv'))
    slots = list(slots_offered['slotsOffered'])

    # get capacities sums
    summary_df['capacity_sum_global'] = summary_df.loc[:, slots[0] + '_Capacity': slots[-1] + '_Capacity'].fillna(
        0).sum(axis=1)
    for day in range(7):
        slot_start = slot_end if day != 0 else list(summary_df.columns).index(slots[0] + '_Capacity')
        slot_end = slot_start + slots_per_day[day]
        summary_df['capacity_sum_day_' + str(day)] = summary_df.iloc[:, slot_start: slot_end].fillna(0).sum(axis=1)

    df_slots = pd.DataFrame()
    for n in tqdm(slots + ['NO_PURCHASE']):
        features = pd.DataFrame()
        features['primary_key'] = summary_df['primary_key']
        features['arrival'] = summary_df['ARRIVAL_DAY']
        features['cut'] = summary_df['ARRIVAL_CAT']
        features['slot'] = n
        features['day'] = n[0]
        if n != 'NO_PURCHASE':
            features['capacity'] = summary_df[n + '_Capacity']
            features['discount'] = summary_df[n + '_Discount']
            features['eco'] = summary_df[n + '_Eco']
            features['capacity_avg_global'] = (summary_df['capacity_sum_global'] - summary_df[n + '_Capacity']) / (
                    len(slots) - 1)
            features['capacity_avg_day'] = (summary_df['capacity_sum_day_' + str(n[0])] - summary_df[
                n + '_Capacity']) / (slots_per_day[int(n[0])] - 1)
        else:
            features['day'] = n
            features['capacity'] = 1
            features['discount'] = 0
            features['eco'] = 0
            features['capacity_avg_global'] = 1
            features['capacity_avg_day'] = 1
        features['order'] = summary_df[n]
        features['avail'] = summary_df['C' + n]

        df_slots = df_slots.append(features)
    df_slots = u.col_to_one_hot(df_slots, 'slot', prefix='slot')
    if save:
        df_slots.to_csv(os.path.join(zone_path, zone_file_prefix + 'unstacked.csv'))
    return df_slots


def unstack_summary_df_ranked(summary_df, zone, root='data', check_saved=False, save=False):  # do for summary
    """
    Unstacks the summary dataframe
    Args:
        summary_df (dataframe): summary dataframe

    Returns: unstacked dataframe
    """
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    if os.path.exists(os.path.join(zone_path, 'RankData', zone_file_prefix + 'unstacked.csv')) and check_saved:
        return pd.read_csv(os.path.join(zone_path, 'RankData', zone_file_prefix + 'unstacked.csv'), index_col=0)
    summary_df['primary_key'] = summary_df['EVENT_DTM'].astype(str) + '-' + summary_df['CUSTOMER_ID'].astype(str)
    slots_per_day = u.get_slots_per_day_for_zone_ranked(zone, root)
    slots, cslots = u.get_slots_observed_ranked(zone_path, zone_file_prefix, summary_df)

    # get capacities sums
    summary_df['capacity_sum_global'] = summary_df.loc[:, slots[0] + '_Capacity': slots[-1] + '_Capacity'].fillna(
        0).sum(axis=1)
    for day in range(7):
        slot_start = slot_end if day != 0 else list(summary_df.columns).index(slots[0] + '_Capacity')
        slot_end = slot_start + slots_per_day[day+1]
        summary_df['capacity_sum_day_' + str(day+1)] = summary_df.iloc[:, slot_start: slot_end].fillna(0).sum(axis=1)
    summary_df = summary_df.fillna(0)
    df_slots = pd.DataFrame()
    for n in tqdm(slots + ['NO_PURCHASE']):
        features = pd.DataFrame()
        features['primary_key'] = summary_df['primary_key']
        features['arrival'] = summary_df['ARRIVAL_DAY']
        features['cut'] = summary_df['ARRIVAL_CAT']
        features['slot'] = n
        features['day'] = n[0]
        if n != 'NO_PURCHASE':
            features['capacity'] = summary_df[n + '_Capacity']
            features['discount'] = summary_df[n + '_Discount']
            features['eco'] = summary_df[n + '_Eco']
            features['capacity_avg_global'] = (summary_df['capacity_sum_global'] - summary_df[n + '_Capacity']) / (
                    len(slots) - 1)
            features['capacity_avg_day'] = (summary_df['capacity_sum_day_' + str(n[0])] - summary_df[n + '_Capacity']) / (
                    slots_per_day[int(n[0])] - 1)
        else:
            features['day'] = n
            features['capacity'] = 1
            features['discount'] = 0
            features['eco'] = 0
            features['capacity_avg_global'] = 1
            features['capacity_avg_day'] = 1
        features['order'] = summary_df[n]
        features['avail'] = summary_df['C' + n]
        df_slots = df_slots.append(features)
    df_slots = u.col_to_one_hot(df_slots, 'slot', prefix='slot', delete=False)
    if save:
        df_slots.to_csv(os.path.join(zone_path, 'RankData', zone_file_prefix + 'unstacked.csv'))
    return df_slots


def stack_df(df, save=True, gr=False):
    slot_columns = [c for c in df.columns if c[:5] == 'slot_']

    df['slot'] = df.loc[:, slot_columns].idxmax(1)
    df['slot'] = df['slot'].apply(lambda x: x[5:])
    df = df.drop(slot_columns, axis=1)
    stacked_df = []
    for name, group in tqdm(df.groupby(['primary_key', 'arrival', 'cut'])):
        stacked_row = {'primary_key': name[0],
                         'cut': name[2],
                         'arrival': name[1]}
        for row in group.iterrows():
            stacked_row[row[1]['slot']] = row[1]['order']
            stacked_row['C_' + row[1]['slot']] = row[1]['avail']
            stacked_row[row[1]['slot'] + '_Capacity'] = row[1]['capacity']
            stacked_row[row[1]['slot'] + '_Eco'] = row[1]['eco']
            if gr:
                stacked_row[row[1]['slot'] + 'Gr'] = row[1]['gr']
            stacked_row[row[1]['slot'] + '_Discount'] = row[1]['discount']
        stacked_df.append(stacked_row)
    stacked_df = pd.DataFrame(stacked_df)
    if save:
        stacked_df.to_csv(os.path.join(results_path, 'stacked.csv'))
    return pd.DataFrame(stacked_df)


def stack_df_ranked(df, gr=False):
    slot_columns = [c for c in df.columns if c[:5] == 'slot_']

    df['slot'] = df.loc[:, slot_columns].idxmax(1)
    df['slot'] = df['slot'].apply(lambda x: x[5:])
    df = df.drop(slot_columns, axis=1)
    stacked_df = []
    for name, group in tqdm(df.groupby(['primary_key', 'arrival', 'cut'])):
        stacked_row = {'primary_key': name[0],
                       'cut': name[2],
                       'arrival': name[1]}
        for row in group.iterrows():
            stacked_row[row[1]['slot']] = row[1]['order']
            stacked_row['C_' + row[1]['slot']] = row[1]['avail']
            stacked_row[row[1]['slot'] + '_Capacity'] = row[1]['capacity']
            stacked_row[row[1]['slot'] + '_Eco'] = row[1]['eco']
            if gr:
                stacked_row[row[1]['slot'] + 'Gr'] = row[1]['gr']
            stacked_row[row[1]['slot'] + '_Discount'] = row[1]['discount']
        stacked_df.append(stacked_row)
    stacked_df = pd.DataFrame(stacked_df)
    # stacked_df.to_csv('stacked.csv')
    return pd.DataFrame(stacked_df)


def stack_df_2(df, save=True):
    slot_columns = [c for c in df.columns if c[:5] == 'slot_']

    df['slot'] = df.loc[:, slot_columns].idxmax(1)
    df['slot'] = df['slot'].apply(lambda x: x[5:])
    df = df.drop(slot_columns, axis=1)
    stacked_df = []
    for name, group in tqdm(df.groupby(['primary_key', 'arrival', 'cut'])):
        stacked_row = {'primary_key': name[0],
                         'cut': name[2],
                         'arrival': name[1]}
        for row in group.iterrows():
            stacked_row[row[1]['slot']] = row[1]['order']
            stacked_row['C_' + row[1]['slot']] = row[1]['avail']
            stacked_row[row[1]['slot'] + '_Capacity'] = row[1]['capacity']
            stacked_row[row[1]['slot'] + '_Eco'] = row[1]['eco']
            stacked_row[row[1]['slot'] + '_Discount'] = row[1]['discount']
        stacked_df.append(stacked_row)
    stacked_df = pd.DataFrame(stacked_df)
    if save:
        stacked_df.to_csv(os.path.join(results_path, 'stacked.csv'))
    return pd.DataFrame(stacked_df)


if __name__ == "__main__":
    # zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    # df = pd.DataFrame()
    # path = os.path.join(zone_path, 'RankData', zone_file_prefix + 'Arrival_Day_{}' + '_Summary.csv')
    # for days in range(7):
    #     df = df.append(pd.read_csv(path.format(days)))
    # print(df)
    # print(unstack_summary_df_ranked(df, zone, root, check_saved=False))
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    summary_df = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'Summary.csv'))
    print(summary_df)
    unstacked_summary_df = unstack_summary_df(summary_df, zone=zone, check_saved=False).dropna(subset=['capacity'])
    print(unstacked_summary_df)
    temp_unstacked = stack_df_2(unstacked_summary_df)
    print(temp_unstacked)
