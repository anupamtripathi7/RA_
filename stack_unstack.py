import pandas as pd
from tqdm import tqdm
import os
import utils as u
from utils import get_zone_output_path


zone = '500.0'
root = 'data'


def unstack_summary_df(summary_df, root='data'):  # do for summary
    """
    Unstacks the summary dataframe
    Args:
        summary_df (dataframe): summary dataframe

    Returns: unstacked dataframe
    """
    summary_df['primary_key'] = summary_df['EVENT_DTM'].astype(str) + '-' + summary_df['CUSTOMER_ID'].astype(str)
    slots_per_day = u.get_slots_per_day_for_zone(zone, root)
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    print(zone_path, zone_file_prefix)
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
    for n in tqdm(slots):
        features = pd.DataFrame()
        features['primary_key'] = summary_df['primary_key']
        features['arrival'] = summary_df['ARRIVAL_DAY']
        features['cut'] = summary_df['ARRIVAL_CAT']
        features['slot'] = n
        features['day'] = n[0]
        features['capacity'] = summary_df[n + '_Capacity']
        features['discount'] = summary_df[n + '_Discount']
        features['eco'] = summary_df[n + '_Eco']
        features['order'] = summary_df[n]
        features['avail'] = summary_df['C' + n]
        features['capacity_avg_global'] = (summary_df['capacity_sum_global'] - summary_df[n + '_Capacity']) / (
                    len(slots) - 1)
        features['capacity_avg_day'] = (summary_df['capacity_sum_day_' + str(n[0])] - summary_df[n + '_Capacity']) / (
                    slots_per_day[int(n[0])] - 1)
        df_slots = df_slots.append(features)
    df_slots = u.col_to_one_hot(df_slots, 'slot', prefix='slot')
    return df_slots


def stack_df(df):
    slot_columns = [c for c in df.columns if c[:5] == 'slot_']

    df['slot'] = df.loc[:, slot_columns].idxmax(1)
    df['slot'] = df['slot'].apply(lambda x: x[5:])
    df = df.drop(slot_columns, axis=1)
    unstacked_df = []
    for name, group in tqdm(df.groupby(['primary_key', 'arrival', 'cut'])):
        unstacked_row = {'primary_key': name[0],
                         'cut': name[2],
                         'arrival': name[1]}
        for row in group.iterrows():
            unstacked_row[row[1]['slot']] = row[1]['order']
            unstacked_row['C_' + row[1]['slot']] = row[1]['avail']
            unstacked_row[row[1]['slot'] + '_Capacity'] = row[1]['capacity']
            unstacked_row[row[1]['slot'] + '_Eco'] = row[1]['eco']
            unstacked_row[row[1]['slot'] + '_Discount'] = row[1]['discount']
        unstacked_df.append(unstacked_row)
    unstacked_df = pd.DataFrame(unstacked_df)
    unstacked_df.to_csv('unstacked.csv')
    return pd.DataFrame(unstacked_df)


if __name__ == "__main__":
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    summary_df = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'Summary.csv'))
    print(summary_df)
    unstacked_summary_df = unstack_summary_df(summary_df).dropna(subset=['capacity'])
    print(unstacked_summary_df)
    # temp_unstacked = stack_df(unstacked_summary_df)
    # print(temp_unstacked)

