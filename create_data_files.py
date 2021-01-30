import pandas as pd
import os
from utils import get_zone_output_path, load_data_file_for_zone, get_slots_observed, get_slots_active, summarize, plot_arrivals, read_offered_slots, get_cut_path, get_hour_list
import matplotlib.pyplot as plt
import utils as u
import numpy as np


data_path = 'data'


def create_summary_by_zone(zones):
    """
    Creates summary file in zone folder
    Args:
        zones (iterable): Zone number
    """

    for zone in zones:
        zone_path, zone_file_prefix = get_zone_output_path(zone, data_path)
        df_avail, df_order, df_steer, df_cap = load_data_file_for_zone(zone, data_path)
        slots_observed, cslots_observed = get_slots_observed(zone_path, zone_file_prefix, df_avail, df_order.columns)
        slots_active, cslots_active, slots_offered, cslots_offered = get_slots_active(zone_path, zone_file_prefix,
                                                                                      df_order, slots_observed)
        summary = summarize(zone_path, zone_file_prefix, df_avail, df_order, df_steer, df_cap,
                            slots_offered, cslots_offered)
        summary = summary.reset_index()
        summary['EVENT_DTM'] = pd.to_datetime(summary['EVENT_DTM'])
        plot_arrivals(zone_path, zone_file_prefix, summary, '60min', 'mean', 'ALL')
        # check to make sure there are no overlapping slots!!!


def create_eco_and_discount_from_steering(zone):
    """
    Reads steering file and creates eco and discount files from it
    Args:
        zone (str): Zone number in float

    Returns:
        (pd.DataFrame): eco_df, discount_df
    """
    zone_path, zone_file_prefix = get_zone_output_path(zone, data_path)
    df_avail, df_order, df_steer, df_cap = load_data_file_for_zone(zone, data_path)


def split_summary_by_day_and_cut():
    """

    Returns:

    """
    zlist = ['700.0', '500.0']
    for zone in zlist:
        zone_path, zone_file_prefix = get_zone_output_path(zone, data_path)
        summary_zone = pd.read_csv(os.path.join(zone_path, 'Zone_' + zone[:-2] + '_Summary.csv'))
        summary_zone['EVENT_DTM'] = pd.to_datetime(summary_zone['EVENT_DTM'])
        slots_zone_offered, cslots_zone_offered = read_offered_slots(zone_path)
        for day in ['0', '1', '2', '3', '4', '5', '6']:
            plot_arrivals(os.path.join(zone_path, day), os.path.join(zone_file_prefix, '_Arrival_Day_' + day),
                          summary_zone, '60min', 'mean', day)

            # save offered slots for each day
            # summary df by day

            for cutcat in ['BEFORE_CUT1', 'BEFORE_CUT2', 'MISSED_BOTH_CUTS']:
                print(zone, day, cutcat)
                cut_path, cut_prefix = get_cut_path(day, cutcat, zone_path, zone_file_prefix)
                summary = summary_zone.loc[
                    (summary_zone['ARRIVAL_CAT'] == cutcat) & (summary_zone['ARRIVAL_DAY'] == int(day))]
                slots_active, cslots_active, slots_offered, cslots_offered = get_slots_active(cut_path, cut_prefix,
                                                                                              summary[slots_zone_offered + ['SLOT_CHOICE']],
                                                                                              slots_zone_offered)
                cols = summary.columns[0:16].tolist() + ['SLOTS_AVAILABLE', 'NO_PURCHASE'] + slots_offered + \
                       ['CNO_PURCHASE'] + cslots_offered + [col + '_Eco' for col in slots_offered] + \
                       [col + '_Discount' for col in slots_offered] + [col + '_Capacity' for col in slots_offered]
                summary[cols].to_csv(os.path.join(cut_path, cut_prefix + 'Summary.csv'), index=False)
                del slots_offered, cslots_offered, slots_active, cslots_active,
                if pd.size(summary):
                    plot_arrivals(cut_path, cut_prefix, summary[cols], '60min', 'mean', day)
                    plt.close()
                del summary, cut_path, cut_prefix


def create_rank_data_by_zone(zone, computername):
    slots_offered_rank = []
    location, filename = get_zone_output_path(zone, computername)
    hourlist = get_hour_list(zone)
    summary = pd.read_csv(os.path.join(location, filename + 'Summary.csv'))
    slots_offered, _ = u.get_slots_active_ranked(location, filename, summary)
    for i in range(1, 8):
        for j in list(hourlist.values()):
            slots_offered_rank = np.append(slots_offered_rank, str(i)+'_'+j)
    summaries = []
    for day in ['0', '1', '2', '3', '4', '5', '6']:
        rank_folder = os.path.join(location, 'RankData')
        rank_file_prefix = filename + 'Arrival_Day_{}_Summary.csv'.format(day)
        summary_day = summary[summary['ARRIVAL_DAY'] == int(day)]
        if day == '1':
            daylist = {'2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '0': '6', '1': '7'}
        elif day == '2':
            daylist = {'2': '7', '3': '1', '4': '2', '5': '3', '6': '4', '0': '5', '1': '6'}
        elif day == '3':
            daylist = {'2': '6', '3': '7', '4': '1', '5': '2', '6': '3', '0': '4', '1': '5'}
        elif day == '4':
            daylist = {'2': '5', '3': '6', '4': '7', '5': '1', '6': '2', '0': '3', '1': '4'}
        elif day == '5':
            daylist = {'2': '4', '3': '5', '4': '6', '5': '7', '6': '1', '0': '2', '1': '3'}
        elif day == '6':
            daylist = {'2': '3', '3': '4', '4': '5', '5': '6', '6': '7', '0': '1', '1': '2'}
        elif day == '0':
            daylist = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '0': '7', '1': '1'}
        summary_day = summary_day.rename(columns=lambda x: daylist[x[0]] + '_' + hourlist[x[2:15]] + x[15:] if x in [col for col in slots_offered] + [col + '_Eco' for col in slots_offered] + [col + '_Discount' for col in slots_offered] + [col + '_Capacity' for col in slots_offered] else x)
        summary_day = summary_day.rename(columns=lambda x: x[0] + daylist[x[1]] + '_' + hourlist[x[3:16]] + x[16:] if x in ['C' + col for col in slots_offered] else x)
        summary_day['SLOT_CHOICE'] = summary_day['SLOT_CHOICE'].fillna('NO_PURCHASE')
        summary_day['SLOT_CHOICE'] = summary_day['SLOT_CHOICE'].apply(lambda row: daylist[row[0]] + '_' + hourlist[row[2:15]] + row[15:] if row in [col for col in slots_offered] else row)
        summary_day.to_csv(os.path.join(rank_folder, filename + 'Arrival_Day_{}_Summary.csv'.format(day)), index=False)
        summaries.append(summary_day)
    summary = pd.concat(summaries, axis=0, sort=False)
    print(slots_offered, list(summary.columns))
    summary = summary[['EVENT_DTM', 'CUSTOMER_ID', 'ORDER_ID', 'ARRIVAL_CAT', 'ARRIVAL_DAY', 'DAY_OF_ORDER', 'CUT_OFF_1', 'CUT_OFF_2', 'SLOT_CHOICE', 'SLOT_STAMP', 'WINDOWS_STEERING', 'ZONE', 'ESTIMATED_SUBTOTAL', 'TOTAL_TIMESLOTS', 'CENSORED', 'TOTAL_ORDER']+[col for col in np.append('NO_PURCHASE', slots_offered_rank)]+['C'+col for col in np.append('NO_PURCHASE', slots_offered_rank)]+[col+'_Eco' for col in slots_offered_rank]+[col+'_Discount' for col in slots_offered_rank] + [col+'_Capacity' for col in slots_offered_rank]]
    summary['CUT2'] = summary['ARRIVAL_CAT'].apply(lambda x: 1 if x == 'BEFORE_CUT2' else 0)
    for day in range(0, 7):
        summary[str(day)+'_ARRIVAL'] = summary['ARRIVAL_DAY'].apply(lambda x: 1 if x == day else 0)
    summary.to_csv(os.path.join(rank_folder, filename + 'SummaryNew.csv'), index=False)


if __name__ == "__main__":
    create_rank_data_by_zone('500.0', data_path)
