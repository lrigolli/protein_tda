import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from data_scripts.fourier_expansion import evaluate_fourier_series


def compute_landscapes_l1_distance(land1_enc, land2_enc, diagnostic=False):
    # land1_enc is dictionary having 'begin', 'end', 'a0','a1',...'an','b0', 'b1', ..., 'bn' as keys.
    # The first two are float values (start, end point of domain and Fourier coefficients of nonzero landscape)

    # Get domain on which at least one of the landscapes is non zero
    x = np.linspace(np.min([land1_enc['begin'], land2_enc['begin']]), np.max([land1_enc['end'], land2_enc['end']]),
                    128 + 1)

    # Define both landscapes values on common set of equispaced points
    x1_num_zero_left = sum(x < land1_enc['begin'])
    x1_num_zero_right = sum(x > land1_enc['end'])
    x1_nonzero = x[x1_num_zero_left:len(x) - x1_num_zero_right]
    L1 = np.abs(x1_nonzero[0] - x1_nonzero[-1]) / 2
    del land1_enc['begin']
    del land1_enc['end']
    y1_nonzero = evaluate_fourier_series(x=x1_nonzero, L=L1, fourier_coefs=land1_enc)
    y1 = np.concatenate([np.array([0] * x1_num_zero_left), y1_nonzero, np.array([0] * x1_num_zero_right)])

    x2_num_zero_left = sum(x < land2_enc['begin'])
    x2_num_zero_right = sum(x > land2_enc['end'])
    x2_nonzero = x[x2_num_zero_left:len(x) - x2_num_zero_right]
    L2 = np.abs(x2_nonzero[0] - x2_nonzero[-1]) / 2
    del land2_enc['begin']
    del land2_enc['end']
    y2_nonzero = evaluate_fourier_series(x=x2_nonzero, L=L2, fourier_coefs=land2_enc)
    y2 = np.concatenate([np.array([0] * x2_num_zero_left), y2_nonzero, np.array([0] * x2_num_zero_right)])

    # L^1 distance is integral of absolute value of difference between y1 and y2
    y_diff = np.abs(y1 - y2)

    # Romberg method for integration is more accurate than trapezoid or Simpson when function is evaluated on 2^k + 1
    # equispaced points
    if np.log2(len(x) - 1) % 1 == 0:
        l1_dist = integrate.romb(y_diff, dx=x[1] - x[0])
    else:
        l1_dist = integrate.simpson(y_diff, x)

    if diagnostic:
        fig = plt.figure()
        plt.plot(x, y1, label='landscape1')
        plt.plot(x, y2, label='landscape2')
        plt.plot(x, y_diff, label='|landscape1-landscape2|')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('landscapes l1 distance')
        plt.legend()
        fig.show()

    return l1_dist


def pairwise_l1_dist(df, pdb_idxs):
    df = df.reset_index(drop=True)
    num_samples = df.shape[0]
    m_dist_half = np.zeros([num_samples, num_samples])
    for i in range(num_samples):
        for j in range(num_samples):
            if j < i:
                m_dist_half[i, j] = compute_landscapes_l1_distance(land1_enc=dict(df.loc[i]), land2_enc=dict(df.loc[j]))
    m_dist = m_dist_half + np.transpose(m_dist_half)
    df_pair_dist = pd.DataFrame(data=m_dist, index=pdb_idxs, columns=pdb_idxs)
    return df_pair_dist


def pairwise_l1_dist_dim(df, dim, grouped_feat_var):
    pairwise_l1_dist_dict = dict()
    for dim_land in [f'dim{dim}_land1', f'dim{dim}_land2', f'dim{dim}_land3']:
        df_dim_land = df[grouped_feat_var['features_var_dict_dim_land'][dim_land]]
        df_dim_land.columns = [col.split('_')[0][:-1] for col in df_dim_land.columns]
        df_pair_l1dist_land_dim = pairwise_l1_dist(df=df_dim_land, pdb_idxs=df['pdb_id'].values)
        pairwise_l1_dist_dict[dim_land] = df_pair_l1dist_land_dim
    return pairwise_l1_dist_dict


def get_discrete_pair_sim(df_cluster):
    all_ids = df_cluster['pdb_ids'].sum()
    num_ids = len(all_ids)
    num_existing_ids = 0
    blocks = []
    for lab, ids in zip(df_cluster['label'], df_cluster['pdb_ids']):
        if lab == -1:
            block = [np.zeros([len(ids), num_existing_ids]),
                     np.eye(len(ids)),
                     np.zeros([len(ids), num_ids-num_existing_ids-len(ids)])]
        else:
            block = [np.zeros([len(ids), num_existing_ids]),
                     np.ones([len(ids), len(ids)]),
                     np.zeros([len(ids), num_ids-num_existing_ids-len(ids)])]
        num_existing_ids += len(ids)
        blocks.append(block)
    df_pair_sim = pd.DataFrame(data=np.block(blocks), index=all_ids, columns=all_ids)
    return df_pair_sim


def convert_clustering_matrix_to_df(df):
    df_clust_dup = df.apply(lambda x: list(np.where(x == 2)[0]), axis=1)
    dict_idx_pdb = dict(enumerate(list(df.columns)))
    # Get list of clusters
    clusters = []
    for clust in df_clust_dup.values:
        clust_pdb = [dict_idx_pdb[el] for el in clust]
        if clust_pdb not in clusters:
            clusters.append(clust_pdb)
    # Store clusters in df
    df_clust = pd.DataFrame(enumerate(clusters), columns=['label', 'cluster'])
    df_clust['num_el'] = df_clust['cluster'].apply(lambda x: len(x))
    df_clust = df_clust.sort_values(by=['num_el'], ascending=False)
    return df_clust
