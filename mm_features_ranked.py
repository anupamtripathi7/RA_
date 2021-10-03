import numpy as np
import pandas as pd
from utils import get_zone_output_path
import os
from scipy.sparse import csr_matrix
from sklearn.utils import graph_shortest_path
import sys
from itertools import combinations


zone = '500.0'
cap_mode = 1                    # 2 for all
# cut = -1                      # -1 for no choice
# arrival = -1                  # -1 for no choice
root = 'data'
results_path = 'results/regression'


def get_active_features(summary_df, slots_offered):                         # prev -> getActiveFeatures
    """
    Gets dataframe with only active discount, eco and gr columns. Also adds constant columns.
    Args:
        summary_df (pd.DataFrame): Summary DataFrame
        slots_offered (list): List of all offered slots
    """
    disc_cols = [col+'_Discount' for col in slots_offered]
    eco_cols = [col+'_Eco' for col in slots_offered]
    gr_cols = [col+'Gr' for col in slots_offered]
    features = summary_df.loc[:, disc_cols+eco_cols+gr_cols]
    features.loc[:, gr_cols] = features.loc[:, gr_cols].abs()
    features = features.loc[:, features.sum(axis=0) != 0]
    disc_cols = set(disc_cols) & set(features.columns)
    eco_cols = set(eco_cols) & set(features.columns)
    gr_cols = set(gr_cols) & set(features.columns)
    for i in reversed(['NO_PURCHASE']+slots_offered):
        features.insert(0, i+'_Asc', value=1)
    return features, list(disc_cols), list(eco_cols), list(gr_cols)


def get_design_matrix(feature_columns, slots_offered):                           # prev -> getDesignMatrix
    """
    Generates a 2d matrix with all features for all slots
    Args:
        feature_columns (list): List of feature columns
        slots_offered (list): List of all offered slots
    """
    df = pd.DataFrame(columns=feature_columns, index=list(np.append(['NO_PURCHASE'], slots_offered)))
    for i in np.append(['NO_PURCHASE'], slots_offered):
        df.loc[i, [col for col in feature_columns if i in col]] = 1
    return df.fillna(0)


def expand_beta(beta, len_disc, len_eco, len_gr):                          # prev -> formBetaExtNumpy
    """
    Expands discount and eco parts of beta for each time slot
    Args:
        beta (np.array): Beta coefficients of length (1 + n_slots + 2)
        len_disc (int): Number of discount columns
        len_eco (int): Number of Eco columns
        len_gr (int): Number of Gr columns

    Returns:
        (np.array): Expanded numpy array
    """
    beta_ext_asc = beta[0:-3][:]
    beta_disc = beta[-3]
    beta_eco = beta[-2]
    beta_gr = beta[-1]
    beta_ext_disc = beta_disc*np.ones(len_disc)
    beta_ext_eco = beta_eco*np.ones(len_eco)
    beta_ext_gr = beta_gr*np.ones(len_gr)
    beta_ext_new = np.concatenate((beta_ext_asc, beta_ext_disc, beta_ext_eco, beta_ext_gr))
    return beta_ext_new


def update_beta(design, features_df, discount_slots, eco_slots, gr_slots, assortment, choice, beta_ext, beta, slots_offered):               # prev => updateBetaNumpy
    beta_new = np.zeros(len(slots_offered)+4)
    features_util = design * beta_ext                   # k*l
    utils = features_df.dot(features_util.T)
    exp_utils = np.exp(utils) * assortment
    Q = exp_utils/exp_utils.sum(axis=1, keepdims=True)
    pred_shares_ext = sum(Q.dot(design)*features_df, 0)
    true_shares_ext = sum(choice.dot(design)*features_df, 0)
    true_share_disc = sum(true_shares_ext[len(slots_offered)+1:len(slots_offered)+1+len(discount_slots)])
    true_share_eco = sum(true_shares_ext[len(slots_offered)+1+len(discount_slots):len(slots_offered)+1+len(discount_slots)+len(eco_slots)])
    true_share_gr = sum(true_shares_ext[len(slots_offered)+1+len(discount_slots)+len(eco_slots):len(slots_offered)+1+len(discount_slots)+len(eco_slots)+len(gr_slots)])
    pred_share_disc = sum(pred_shares_ext[len(slots_offered)+1:len(slots_offered)+1+len(discount_slots)])
    pred_share_eco = sum(pred_shares_ext[len(slots_offered)+1+len(discount_slots):len(slots_offered)+1+len(discount_slots)+len(eco_slots)])
    pred_share_gr = sum(pred_shares_ext[len(slots_offered)+1+len(discount_slots)+len(eco_slots):len(slots_offered)+1+len(discount_slots)+len(eco_slots)+len(gr_slots)])
    true_share = np.append(true_shares_ext[:len(slots_offered)+1], [true_share_disc, true_share_eco, true_share_gr])
    pred_share = np.append(pred_shares_ext[:len(slots_offered)+1], [pred_share_disc, pred_share_eco, pred_share_gr])
    beta_new[1:] = beta[1:][:] + np.log(true_share[1:][:]) - np.log(pred_share[1:][:])
    if not discount_slots:
        beta_new[-3] = 0
    if not eco_slots:
        beta_new[-2] = 0
    if not gr_slots:
        beta_new[-1] = 0
    return [beta_new, Q]


# def mm_features(Location, filename, summary, slots_offered):
#     beta_coef = np.append([0], np.random.rand(len(slots_offered) + 3))
#     features_df, disc_cols, eco_cols, gr_cols = get_active_features(summary, slots_offered)
#     beta_ext = expand_beta(beta_coef, len(disc_cols), len(disc_cols), len(gr_cols))
#     design_df = get_design_matrix(features_df.columns.tolist(), slots_offered)
#     assortment_df = summary.loc[:, ['C' + col for col in ['NO_PURCHASE'] + slots_offered]].fillna(0)
#     choice_df = summary.loc[:, [col for col in ['NO_PURCHASE'] + slots_offered]].fillna(0)
#     design = design_df.values
#     features = features_df.values
#     assortment = assortment_df.values
#     choice = choice_df.values
#
#     i = 0
#     while True:
#         i += 1
#         beta = np.copy(beta_coef)
#         beta_ext_cp = np.copy(beta_ext)
#         beta_coef, Q = update_beta(design, features, disc_cols, eco_cols, gr_cols, assortment, choice, beta_ext_cp, beta, slots_offered)
#         log_likeli = sum(np.log(sum(Q * choice, 1)))
#         beta_ext = expand_beta(beta_coef, len(disc_cols), len(disc_cols), len(gr_cols))
#         print('Iteration=', i, 'loglikelihood =', log_likeli, 'beta_disc', beta_coef[-3], 'beta_eco', beta_coef[-2], 'beta_gr', beta_coef[-1])
#         if np.linalg.norm(beta_coef - beta) < 10 ** -6 or i > 500:
#             predict_prob_df = pd.DataFrame(Q, columns=['NO_PURCHASE'] + slots_offered)
#             beta_df = pd.DataFrame([np.array(beta_coef)], columns=['NO_PURCHASE'] + slots_offered + ['Discount', 'Eco', 'Gr'])
#             predict_prob_df.to_csv(Location + filename + 'predprobfeatures.csv')
#             beta_df.to_csv(Location + filename + 'betafeatures.csv')
#             del summary, predict_prob_df, design_df, features_df, assortment_df, choice_df, design, features, assortment, choice
#             break
#     return beta_df.iloc[0]


def MMfeaturesBoot(Location, filename, summary, slots_offered):
    beta_coef = np.append([0], np.random.rand(len(slots_offered) + 3))
    features_df, disc_cols, eco_cols, gr_cols = get_active_features(summary, slots_offered)
    beta_ext = expand_beta(beta_coef, len(disc_cols), len(eco_cols), len(gr_cols))
    design_df = get_design_matrix(features_df.columns.tolist(), slots_offered)
    assortment_df = summary.loc[:, ['C_' + col for col in ['NO_PURCHASE'] + slots_offered]].fillna(0)
    choice_df = summary.loc[:, [col for col in ['NO_PURCHASE'] + slots_offered]].fillna(0)
    design = design_df.values
    features = features_df.values
    assortment = assortment_df.values
    choice = choice_df.values

    C = np.where(choice == 1)[1]
    membership = assortment
    nprods = assortment.shape[1]
    ## check if the MM algorithm would coverge by testing if the item-item graph
        # is strongly connected
    row = []
    col = []
    data = []
    for i in range(membership.shape[0]):
        assort = list(np.nonzero(membership[i, :])[0])
        try:
            assort.remove(C[i])
        except ValueError:
            print (i, C[i], assort)
            break
        row += len(assort)*[C[i]]
        col += assort
        data += len(assort)*[1]

    dist_matrix = csr_matrix( (data, (row, col)), shape=(nprods, nprods) )
    Z = graph_shortest_path.graph_shortest_path(dist_matrix, method='D') # Dijkstra's algorithm
    I = np.eye(nprods)
    if np.count_nonzero(I+Z) < nprods**2:
        # condition for convergence of MM algo not met
        sys.stderr.write('Warning: Convergence condition for MM algorithm not met...adding noise to the data matrix...\n')
        pairs = [pair for pair in combinations(np.delete(np.arange(nprods),0), 2)]
        npairs = len(pairs)
        pairs = np.array(pairs)
        pairs = np.tile(pairs, (2, 1))
        Z = np.zeros((len(pairs), nprods))
        for i,pair in enumerate(pairs): Z[i, pair] = 1
        assortment = np.vstack((assortment, Z))
        d = np.append(pairs[:npairs, 0],pairs[npairs:, 1])
        choicenew=np.zeros((Z.shape[0],nprods))
        choicenew[np.arange(Z.shape[0]),d] = 1
        choice = np.vstack((choice, choicenew))
        featuresnew=np.zeros((Z.shape[0], features.shape[1]))
        featuresnew[:, np.arange(nprods)] = 1
        features = np.vstack((features, featuresnew))
    i = 0
    while True:
        i += 1
        beta = np.copy(beta_coef)
        beta_ext_cp = np.copy(beta_ext)
        beta_coef, Q = update_beta(design, features, disc_cols, eco_cols, gr_cols, assortment, choice, beta_ext_cp,
                                   beta, slots_offered)
        log_likeli = sum(np.log(sum(Q * choice, 1)))
        beta_ext = expand_beta(beta_coef, len(disc_cols), len(eco_cols), len(gr_cols))
        print('Iteration=', i, 'loglikelihood =', log_likeli, 'beta_disc', beta_coef[-3], 'beta_eco', beta_coef[-2],
              'beta_gr', beta_coef[-1])
        if np.linalg.norm(beta_coef[:-1] - beta[:-1]) < 10 ** -6 or i > 500:
            predict_prob_df = pd.DataFrame(Q, columns=['NO_PURCHASE'] + slots_offered)
            beta_df = pd.DataFrame([np.array(beta_coef)],
                                   columns=['NO_PURCHASE'] + slots_offered + ['Discount', 'Eco', 'Gr'])
            predict_prob_df.to_csv(Location + filename + 'predprobfeatures.csv')
            beta_df.to_csv(Location + filename + 'betafeatures.csv')
            del summary, predict_prob_df, design_df, features_df, assortment_df, choice_df, design, features, assortment, choice
            break
    return beta_df.iloc[0]


if __name__ == "__main__":
    zone_path, zone_file_prefix = get_zone_output_path(zone, root)
    summary = pd.read_csv(
        '/Users/anupamtripathi/PycharmProjects/RA_/results/gr/500.0/gr_unranked_stacked_arrival_0_cut_-1_cap_2.csv')
    summary = summary.drop(['NO_PURCHASE_Eco', 'NO_PURCHASE_Discount'], axis=1)
    summary = summary.dropna()
    slots_offered = pd.read_csv(os.path.join(zone_path, zone_file_prefix + 'SlotsOfferedTitle.csv'))
    MMfeaturesBoot(zone_path, zone_file_prefix, summary, slots_offered['slotsOffered'].tolist())
