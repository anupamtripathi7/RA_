import pandas as pd
import os
from utils import get_zone_output_path, load_data_file_for_zone, get_slots_observed, get_slots_active, summarize, plot_arrivals, read_offered_slots, get_cut_path
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    create_summary_by_zone(['500.0'])
