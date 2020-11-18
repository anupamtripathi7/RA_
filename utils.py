import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


def col_to_one_hot(df, col_name, prefix='', delete=True):                                   # new
    """
    Returns a dataframe with the specified column transformed to one hot
    Args:
        df (dataframe): Dataframe to be categorized
        col_name (str): Name of column which needs to be made one hot
        prefix (str): Prefix of all new one hot columns
        delete (bool): If true, deleted the column after forming one hot columns from it
    Returns: dataframe with column in one hot
    """
    one_hot_df = pd.get_dummies(df[col_name], prefix=prefix)
    if delete:
        df = df.drop(col_name, axis=1)
    df = pd.concat([df, one_hot_df.reindex(df.index)], axis=1)
    return df


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


def load_data_file_for_zone(zone, folder):                                                  # prev -> createAvailOrderSteerByZone
    """
    Loads and preprocess avial, order, steering n=and capacity csv files
    Args:
        zone (str): Zone number in float string
        folder (str): Root data folder
    Returns:
        avail, order, steering and capacity files as dataframes
    """
    df_avail, df_order, df_steer, df_cap = read_aggregate_by_zone_files(zone, folder)

    df_avail = df_avail.replace(to_replace=['A', 'H', 'S', 'R', 'E', '#MULTIVALUE'], value=[1, 0, 0, 0, 0, 0])
    df_order['CENSORED'] = df_order['CENSORED'].fillna(0)
    df_order['CENSORED'] = df_order['CENSORED'].astype(int)
    df_order.rename(columns={'1ST_CUTOFF': 'CUT_OFF_1', '2ND_CUTOFF': 'CUT_OFF_2'}, inplace=True)
    
    df_avail = df_avail.drop(df_avail.columns[0], axis=1)
    df_order = df_order.drop(df_order.columns[0], axis=1)
    df_steer = df_steer.drop(df_steer.columns[0], axis=1)
    df_cap = df_cap.drop(df_cap.columns[0], axis=1)

    df_order['CUT_OFF_1'] = pd.to_datetime(df_order['CUT_OFF_1'])
    df_order['CUT_OFF_2'] = pd.to_datetime(df_order['CUT_OFF_2'])
    df_avail.insert(1, 'ARRIVAL_DATE', pd.to_datetime(df_avail['EVENT_DTM'].apply(lambda x: str(x.date()))))
    df_order.insert(1, 'ARRIVAL_DATE', pd.to_datetime(df_order['EVENT_DTM'].apply(lambda x: str(x.date()))))
    df_steer.insert(1, 'ARRIVAL_DATE', pd.to_datetime(df_steer['EVENT_DTM'].apply(lambda x: str(x.date()))))
    df_cap.insert(1, 'ARRIVAL_DATE', pd.to_datetime(df_cap['EVENT_DTM'].apply(lambda x: str(x.date()))))
    df_avail = df_avail.set_index(['EVENT_DTM', 'CUSTOMER_ID'])
    df_order = df_order.set_index(['EVENT_DTM', 'CUSTOMER_ID'])
    df_steer = df_steer.set_index(['EVENT_DTM', 'CUSTOMER_ID'])
    df_cap = df_cap.set_index(['EVENT_DTM', 'CUSTOMER_ID'])

    df_steer = df_steer.rename(
        columns=lambda x: x[1:] if (str(x).startswith('C')) and (str(x).startswith('C')) else x)
    df_cap = df_cap.rename(columns=lambda x: x[1:] + '_Capacity' if (str(x).startswith('C')) else x)

    """
           ##### Output the data #####
    """
    zone_output_path, zone_file_prefix = get_zone_output_path(zone, folder)
    if zone == '700.0':
        df_order['5_06:00 - 08:00'] = df_order['5_06:00 - 08:00'].fillna(0) + df_order['5_06:30 - 08:00'].fillna(0)
        df_order['6_06:00 - 08:00'] = df_order['6_06:00 - 08:00'].fillna(0) + df_order['6_06:30 - 08:00'].fillna(0)
        df_order = df_order.drop('6_05:00 - 06:00', axis=1)
        df_order = df_order.rename(columns={"6_05:00 - 06:30": "6_05:00 - 06:00"})
        df_order = df_order.drop('5_06:30 - 08:00', axis=1)
        df_order = df_order.drop('6_06:30 - 08:00', axis=1)
        df_order["SLOT_CHOICE"].replace({"5_06:30 - 08:00": "5_06:00 - 08:00", "6_06:30 - 08:00": "6_06:00 - 08:00",
                                           "6_05:00 - 06:30": "6_05:00 - 06:00"}, inplace=True)
        df_avail['C5_06:00 - 08:00'] = df_avail['C5_06:00 - 08:00'].fillna(0) + df_avail[
            'C5_06:30 - 08:00'].fillna(0)
        df_avail['C6_06:00 - 08:00'] = df_avail['C6_06:00 - 08:00'].fillna(0) + df_avail[
            'C6_06:30 - 08:00'].fillna(0)
        df_avail = df_avail.drop('C6_05:00 - 06:00', axis=1)
        df_avail = df_avail.rename(columns={"C6_05:00 - 06:30": "C6_05:00 - 06:00"})
        df_avail = df_avail.drop('C5_06:30 - 08:00', axis=1)
        df_avail = df_avail.drop('C6_06:30 - 08:00', axis=1)
        df_cap['5_06:00 - 08:00_Capacity'] = df_cap['5_06:00 - 08:00_Capacity'].fillna(0) + df_cap[
            '5_06:30 - 08:00_Capacity'].fillna(0)
        df_cap['6_06:00 - 08:00'] = df_cap['6_06:00 - 08:00_Capacity'].fillna(0) + df_cap[
            '6_06:30 - 08:00_Capacity'].fillna(0)
        df_cap = df_cap.drop('6_05:00 - 06:00_Capacity', axis=1)
        # df_cap=df_cap.rename(columns={"5_05:00 - 06:30_Capacity":"5_05:00 - 06:00_Capacity"})
        df_cap = df_cap.rename(columns={"6_05:00 - 06:30_Capacity": "6_05:00 - 06:00_Capacity"})
        df_cap = df_cap.drop('5_06:30 - 08:00_Capacity', axis=1)
        df_cap = df_cap.drop('6_06:30 - 08:00_Capacity', axis=1)
        df_steer['5_06:00 - 08:00_Eco'] = df_steer['5_06:00 - 08:00_Eco'].fillna(0) + df_steer[
            '5_06:30 - 08:00_Eco'].fillna(0)
        df_steer['6_06:00 - 08:00_Eco'] = df_steer['6_06:00 - 08:00_Eco'].fillna(0) + df_steer[
            '6_06:30 - 08:00_Eco'].fillna(0)
        df_steer = df_steer.drop('6_05:00 - 06:00_Eco', axis=1)
        df_steer = df_steer.rename(columns={"6_05:00 - 06:30_Eco": "6_05:00 - 06:00_Eco"})
        df_steer = df_steer.drop('5_06:30 - 08:00_Eco', axis=1)
        df_steer = df_steer.drop('6_06:30 - 08:00_Eco', axis=1)
        df_steer['5_06:00 - 08:00_Discount'] = df_steer['5_06:00 - 08:00_Discount'].fillna(0) + df_steer[
            '5_06:30 - 08:00_Discount'].fillna(0)
        df_steer['6_06:00 - 08:00_Discount'] = df_steer['6_06:00 - 08:00_Discount'].fillna(0) + df_steer[
            '6_06:30 - 08:00_Discount'].fillna(0)
        df_steer = df_steer.drop('6_05:00 - 06:00_Discount', axis=1)
        df_steer = df_steer.rename(columns={"6_05:00 - 06:30_Discount": "6_05:00 - 06:00_Discount"})
        df_steer = df_steer.drop('5_06:30 - 08:00_Discount', axis=1)
        df_steer = df_steer.drop('6_06:30 - 08:00_Discount', axis=1)
    df_avail.to_csv(os.path.join(zone_output_path, zone_file_prefix + 'AvailAll.csv'))
    df_order.to_csv(os.path.join(zone_output_path, zone_file_prefix + 'OrderAll.csv'))
    df_steer.to_csv(os.path.join(zone_output_path, zone_file_prefix + 'SteeringAll.csv'))
    df_cap.to_csv(os.path.join(zone_output_path, zone_file_prefix + 'CapAll.csv'))

    return df_avail, df_order, df_steer, df_cap


def read_aggregate_by_zone_files(zone, folder):                                         # new
    """
    Reads avail, order, steering and capacity files from "Aggregate by zone" folder
    Args:
        zone (str): Zone number in float string
        folder (str): Root data folder

    Returns:
        avail, order, steering and capacity files as dataframes
    """
    avail_path = os.path.join(folder, 'Aggregate By Zone', zone + '_Avail.csv')
    order_path = os.path.join(folder, 'Aggregate By Zone', zone + '_Order.csv')
    steer_path = os.path.join(folder, 'Aggregate By Zone', zone + '_Steering.csv')
    cap_path = os.path.join(folder, 'Aggregate By Zone', zone + '_Cap.csv')

    df_avail = pd.read_csv(avail_path, parse_dates=['EVENT_DTM'])
    df_order = pd.read_csv(order_path, parse_dates=['EVENT_DTM'])
    df_steer = pd.read_csv(steer_path, parse_dates=['EVENT_DTM'])
    df_cap = pd.read_csv(cap_path, parse_dates=['EVENT_DTM'])

    return df_avail, df_order, df_steer, df_cap


def get_zone_output_path(zone, folder):                                                                  # prev -> getLocation
    """
    Returns the output folder name and output file prefix for a zone
    Args:
        zone (str): Zone number in float string
        folder (str): Root data folder
    Returns:
        output folder name and output file prefix
    """
    return os.path.join(folder, 'Zones', zone[:-2]), 'Zone_'+zone[:-2] + '_'


def get_slots_observed(zone_folder, filename, df_avail, df_order_columns):                                  # prev -> getSlotsObserved
    """
    Remove time-slots which appear in the Avail sheet but not in the Order sheet
    Args:
        zone_folder (str): Zone output folder path
        filename (str): Zone file name prefix
        df_avail (dataframe): Avail dataframe
        df_order_columns (list): Columns names from df_order

    Returns:
        offered and censored time-slots

    """
    df_offered = df_avail.dropna(axis=1, how='all')
    for i in df_offered.columns:
        if str(i).endswith('0') and i[1:] not in df_order_columns:
            df_offered = df_offered.drop(i, axis=1)
    slots_offered = []
    slots_censored = []
    for i in df_offered.columns:
        if str(i).endswith('0'):
            slots_offered.append(i[1:])
            slots_censored.append(i)
    df = pd.DataFrame()
    df['slotsOffered'] = slots_offered
    df['cslotsOffered'] = slots_censored
    df.to_csv(os.path.join(zone_folder, filename + 'SlotsObservedTitle.csv'), index=False)
    return slots_offered, slots_censored


def get_slots_active(zone_folder, filename, df_order, slots_observed):                         # prev -> getSlotsActive
    """
    Gets the slots that were picked at least once
    Args:
        zone_folder (str): Zone output folder path
        filename (str): Zone file name prefix
        df_order (dataframe): Order dataframe
        slots_observed (list): list of slots offered

    Returns:
        offered and censored time-slots

    """
    slots_active = []
    cslots_active = []
    for i in df_order.columns:
        if i.endswith('0') and (sum(df_order[i]) > 1):
            slots_active.append(i)
            cslots_active.append('C'+i)
    df = pd.DataFrame()
    df['slotsActive'] = slots_active
    df['cslotsActive'] = cslots_active
    df.to_csv(os.path.join(zone_folder, filename + 'SlotsActiveTitle.csv'), index=False)
    k = list(df_order['SLOT_CHOICE'].values)
    slots_offered = sorted(list(set(np.unique(k)) & set(slots_observed)))
    slots_censored = ['C'+i for i in slots_offered]
    df2 = pd.DataFrame()
    df2['slotsOffered'] = slots_offered
    df2['cslotsOffered'] = slots_censored
    df2.to_csv(os.path.join(zone_folder, filename + 'SlotsOfferedTitle.csv'), index=False)
    return slots_active, cslots_active, slots_offered, slots_censored


def summarize(zone_folder, filename, df_avail, df_order, df_steer, df_cap, slots_offered, slots_censored):
    """
    Creates summary file
    Args:
        zone_folder (str): Zone output folder path
        filename (str): Zone file name prefix
        df_avail (dataframe): Avail dataframe
        df_order (dataframe): Order dataframe
        df_steer (dataframe): Steering dataframe
        df_cap (dataframe): Capacity dataframe
        slots_offered (list): List of offered slots
        slots_censored (list): List of censored slots

    Returns:
        (dataframe): summary
    """
    df_order_offered = df_order[['ORDER_ID', 'ARRIVAL_CAT', 'ARRIVAL_DAY', 'DAY_OF_ORDER', 'CUT_OFF_1', 'CUT_OFF_2',
                                 'SLOT_CHOICE', 'SLOT_STAMP', 'WINDOWS_STEERING', 'ZONE', 'ESTIMATED_SUBTOTAL',
                                 'TOTAL_TIMESLOTS', 'CENSORED', 'TOTAL_ORDER', 'NO_PURCHASE'] + slots_offered]
    df_steer = df_steer.loc[:, [col+'_Eco' for col in slots_offered]+[col+'_Discount' for col in slots_offered]]
    df_steer = df_steer.replace(to_replace=[2, np.nan], value=[1, 0])
    df_avail_offered = df_avail[slots_censored]
    df_avail_offered.insert(0, 'CNO_PURCHASE', value=1)
    df_cap = df_cap.loc[:, [col+'_Capacity' for col in slots_offered]]

    # remove duplicate index
    df_avail_offered = df_avail_offered[~df_avail_offered.index.duplicated(keep='first')]
    df_order_offered = df_order_offered[~df_order_offered.index.duplicated(keep='first')]
    df_steer = df_steer[~df_steer.index.duplicated(keep='first')]
    df_cap = df_cap[~df_cap.index.duplicated(keep='first')]
    summary = pd.concat([df_order_offered, df_avail_offered, df_steer, df_cap], axis=1)

    k = summary['SLOT_CHOICE'].values.tolist()
    for i in np.unique(k):
        if i not in slots_offered+['nan', '0']:
            summary = summary[summary['SLOT_CHOICE'] != i]
            print('Attention Removing orders for timeslot ', i, ' which appear in orders but not in Avail - Zone')

    # adjust availability of slots that are chosen but are not available
    kdf = summary[[col for col in slots_offered]].fillna(0)
    kdf2 = summary[['C'+col for col in slots_offered]].fillna(0)
    kdf2[['C'+col for col in slots_offered]] = kdf2.values-kdf.values
    kdf3 = kdf2.replace(to_replace=-1, value=0)
    summary[['C'+col for col in slots_offered]] = kdf3+kdf.values
    summary.loc[summary['SLOT_CHOICE'] == '0', 'SLOT_CHOICE'] = ''
    summary['SLOTS_AVAILABLE'] = summary[slots_censored].sum(axis=1)
    summary.to_csv(os.path.join(zone_folder, filename + 'Summary.CSV'), mode='w')
    return summary


def plot_arrivals(zone_folder, filename, summary, freq, agg_option, day):                       # prev -> plotArrivals
    """

    Args:
        zone_folder (str): Zone output folder path
        filename (str): Zone file name prefix
        summary (dataframe): summary dataframe
        freq (str): frequency with units
        agg_option (str): aggregation method
        day (str): days

    Returns:

    """
    summary = summary.copy()
    summary.insert(1, 'ARRIVAL_IND', 1, True)
    summary['EVENT_DTM'] = pd.to_datetime(summary['EVENT_DTM'])

    new1 = summary[['EVENT_DTM', 'ARRIVAL_IND', 'TOTAL_ORDER', 'CENSORED']].copy()
    new2 = summary[['EVENT_DTM', 'SLOTS_AVAILABLE']].copy()
    new1 = new1.set_index('EVENT_DTM')
    new2 = new2.set_index('EVENT_DTM')

    if day != 'ALL':
        new1 = new1[new1.index.dayofweek == int(day)]
        new2 = new2[new2.index.dayofweek == int(day)]
    new1 = new1.resample('60min').sum()
    new2 = new2.resample('60min').mean()
    new1['DAY'] = new1.index.map(lambda x: x.weekday())
    new1['TIME'] = new1.index.map(lambda x: x.time())
    new2['DAY'] = new1.index.map(lambda x: x.weekday())
    new2['TIME'] = new1.index.map(lambda x: x.time())
    new1 = new1.fillna(0)
    arrivals = new1.pivot_table(values='ARRIVAL_IND', columns='DAY', index='TIME', aggfunc=agg_option)
    arrivals = pd.DataFrame(arrivals.sum(axis=1))
    arrivals.columns = ['No Purchase']
    orders = new1.pivot_table(values='TOTAL_ORDER', columns='DAY', index='TIME', aggfunc=agg_option)
    orders = pd.DataFrame(orders.sum(axis=1))
    orders.columns = ['Placed Order']
    censored = new2.pivot_table(values='SLOTS_AVAILABLE', columns='DAY', index='TIME', aggfunc='mean')
    censored = pd.DataFrame(censored.mean(axis=1))
    censored = censored.reindex(arrivals.index)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    orders.plot(ax=ax, kind='bar', alpha=0.9, color='r', label='Placed Orders')
    arrivals.plot(ax=ax, kind='bar', color='black', alpha=0.3, label='No Purchase')
    ax.set_ylabel('Average Orders and No Purchases', fontsize=15)
    ax2 = ax.twinx()
    ax2.set_ylabel('Average Availability Level', fontsize=15)
    ax2.plot(ax.get_xticks(), censored, marker='o', color='k', linestyle='--', label='Availability Level')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=15)
    ax.set_xlabel("Arrival Time (freq = %s)" % freq)
    plt.title("ARRIVAL_%s Versus Time" % day)
    if not os.path.exists(zone_folder):
        os.makedirs(zone_folder)
    fig.savefig(os.path.join(zone_folder, filename+'ARRIVAL_%s_Versus_Time_freq_%s.pdf' % (day, freq)),
                aspect='auto', bbox_inches='tight', fontsize=15, dpi=600)
    plt.close()


def read_offered_slots(folder):                                                            # prev -> readOfferedSlotsTitle
    """
    Reads slots offered csv
    Args:
        folder (str): path to csv
    Returns:
        (list): Slots offered
        (list): Slots censored
    """
    df = pd.read_csv(os.path.join(folder, '_slotsOfferedTitle.csv'))
    slots_offered = df['slotsOffered'].tolist()
    cslots_offered = df['cslotsOffered'].tolist()
    return [slots_offered, cslots_offered]


def get_cut_path(day, cutcat, zone_path, zone_prefix):                                       # prev -> getLocationCut
    """
    Returns the path to cut folder and file prefix
    Args:
        day (str): day number
        cutcat (str): cut
        zone_path (str): path to zone
        zone_prefix (str): zone file prefix

    Returns:
        path to cut folder and file prefix
    """
    if cutcat in ['BEFORE_CUT1', 'BEFORE_CUT2', 'MISSED_BOTH_CUTS']:
        folder = os.path.join(zone_path, day, cutcat)
        filename = zone_prefix + '_Arrival_Day_' + day + '_' + cutcat + '_'
    else:
        folder = os.path.join(zone_path, day)
        filename = zone_prefix + '_Arrival_Day_' + day + '_'
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder, filename


def get_summary_col_info(summary_df, zone, root='data'):
    """
    Get column start and end number for avail, order, capacity, discount and eco
    Args:
        summary_df (dataframe): summary dataframe
        zone (str): zone number in float
        root (str, optional): root directory path

    Returns:
        (dict): start and end cols for each type
    """
    df_info = {}
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    slots_offered = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'SlotsObservedTitle.csv'))

    col_no = 0
    while col_no < len(summary_df.columns):
        # avail slots
        if summary_df.columns[col_no].startswith('C0_'):
            df_info['avail'] = (col_no, col_no + len(slots_offered))
            col_no += len(slots_offered)
        # avail slots
        elif summary_df.columns[col_no].endswith('_Eco'):
            df_info['eco'] = (col_no, col_no + len(slots_offered))
            col_no += len(slots_offered)
        # discount slots
        elif summary_df.columns[col_no].endswith('_Discount'):
            df_info['discount'] = (col_no, col_no + len(slots_offered))
            col_no += len(slots_offered)
        # capacity slots
        elif summary_df.columns[col_no].endswith('_Capacity'):
            df_info['capacity'] = (col_no, col_no + len(slots_offered))
            col_no += len(slots_offered)
        # order slots
        elif summary_df.columns[col_no].startswith('0_'):
            df_info['order'] = (col_no, col_no + len(slots_offered))
            col_no += len(slots_offered)
        else:
            col_no += 1
    return df_info


def get_slots_per_day_for_zone(zone, root='data'):
    """
    Get the number of slots per day for a zone
    Args:
        zone (str): zone number in float
        root (str, optional): root directory path

    Returns:
        (dict): slots per day

    """
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    slots_offered = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'SlotsObservedTitle.csv'))
    slots_offered['day'] = slots_offered['slotsOffered'].apply(lambda x: x[0])
    return slots_offered.groupby('day').count().slotsOffered.to_dict()


if __name__ == '__main__':
    zone = '700.0'
    data_path = 'data'
    zone_path, zone_file_prefix = get_zone_output_path(zone, data_path)
    df_avail, df_order, df_steer, df_cap = load_data_file_for_zone(zone, data_path)
    slots_observed, cslots_observed = get_slots_observed(zone_path, zone_file_prefix, df_avail, df_order.columns)
    slots_active, cslots_active, slots_offered, cslots_offered = get_slots_active(zone_path, zone_file_prefix,
                                                                                  df_order, slots_observed)