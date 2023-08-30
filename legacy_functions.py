import numpy as np
import pandas as pd


def dist_land_shape(x, y):
    # this is just L^2 distance between the curves described by Fourier coefficients
    return np.linalg.norm(x-y)


def dist_land_begin_end(x, y):
    return np.mean(np.abs(x-y))


def dist_dim0(x, y):
    # compute mean absolute distance. don't use Euclidean distance since don't want it to depend on numner of features
    # used to encode dim0
    return np.mean(np.abs(x-y))


def pairwise_dist(df, dict_features_var_dims_lands):
    # w weight in [0,1] for average land
    # for the moment give same importance to shape and start/end of landscape
    # (do further analysis to determine which weight makes more sense)
    # Compute similarity between pair of samples
    num_samples = df.shape[0]

    pair_dist_dict = {}
    samples_dict = {}
    for key in dict_features_var_dims_lands.keys():
        samples_dict[f"mat_{key}"] = df[dict_features_var_dims_lands[key]].values
        pair_dist_dict[f"pair_dist_{key}"] = np.zeros([num_samples, num_samples])

    for key_sample, key_dist in zip(samples_dict.keys(), pair_dist_dict.keys()):
        m_sample = samples_dict[key_sample]
        m_dist_half = pair_dist_dict[key_dist]
        for i in range(num_samples):
            for j in range(num_samples):
                if j < i:
                    if 'dim0' in key_sample:
                        m_dist_half[i, j] = dist_dim0(m_sample[i, :], m_sample[j, :])
                    elif 'shape' in key_sample:
                        m_dist_half[i, j] = dist_land_shape(m_sample[i, :], m_sample[j, :])
                    elif 'begin_end' in key_sample:
                        m_dist_half[i, j] = dist_land_begin_end(m_sample[i, :], m_sample[j, :])
                    else:
                        m_dist_half[i, j] = m_dist_half[i, j]
        m_dist = m_dist_half + np.transpose(m_dist_half)
        df_pair_dist = pd.DataFrame(data=m_dist, index=df['pdb_id'].values, columns=df['pdb_id'].values)
        pair_dist_dict[key_dist] = df_pair_dist
    return pair_dist_dict


def aggregate_metrics_dim(pair_dist_dict):
    pair_dist_dict_dims = {}
    counter_dim1_shape = 0
    counter_dim1_extrema = 0
    counter_dim2_shape = 0
    counter_dim2_extrema = 0
    for key in pair_dist_dict.keys():
        # aggregate dim 0
        if 'dim0' in key:
            pair_dist_dict_dims['dim0'] = pair_dist_dict[key]
        # aggregate shape in dim 1
        if 'dim1' in key:
            if 'shape' in key:
                # normalize values
                df_normalized_dim1_shape = pair_dist_dict[key] / np.mean(np.mean(pair_dist_dict[key], axis=0))
                if counter_dim1_shape > 0:
                    df_agg_dim1_shape += (1 / 3) * df_normalized_dim1_shape
                else:
                    df_agg_dim1_shape = (1 / 3) * df_normalized_dim1_shape
                    counter_dim1_shape += 1
            if 'begin_end' in key:
                # normalize values
                df_normalized_dim1_extrema = pair_dist_dict[key] / np.mean(np.mean(pair_dist_dict[key], axis=0))
                if counter_dim1_extrema > 0:
                    df_agg_dim1_extrema += (1 / 3) * df_normalized_dim1_extrema
                else:
                    df_agg_dim1_extrema = (1 / 3) * df_normalized_dim1_extrema
                    counter_dim1_extrema += 1

        # aggregate shape in dim 2
        if 'dim2' in key:
            if 'shape' in key:
                # normalize values
                df_normalized_dim2_shape = pair_dist_dict[key] / np.mean(np.mean(pair_dist_dict[key], axis=0))
                if counter_dim2_shape > 0:
                    df_agg_dim2_shape += (1 / 3) * df_normalized_dim2_shape
                else:
                    df_agg_dim2_shape = (1 / 3) * df_normalized_dim2_shape
                    counter_dim2_shape += 1
            if 'begin_end' in key:
                # normalize values
                df_normalized_dim2_extrema = pair_dist_dict[key] / np.mean(np.mean(pair_dist_dict[key], axis=0))
                if counter_dim2_extrema > 0:
                    df_agg_dim2_extrema += (1 / 3) * df_normalized_dim2_extrema
                else:
                    df_agg_dim2_extrema = (1 / 3) * df_normalized_dim2_extrema
                    counter_dim2_extrema += 1

    # aggregate shape and begin_end in dim_1 and 2
    pair_dist_dict_dims['dim1'] = (1 / 2) * (df_agg_dim1_shape + df_agg_dim1_extrema)
    pair_dist_dict_dims['dim2'] = (1 / 2) * (df_agg_dim2_shape + df_agg_dim2_extrema)
    return pair_dist_dict_dims


def aggregate_metrics_all(pair_dist_dict_dims):
    df_normalized_dim0 = pair_dist_dict_dims['dim0']/np.mean(np.mean(pair_dist_dict_dims['dim0'], axis=0))
    return (df_normalized_dim0 + pair_dist_dict_dims['dim1'] + pair_dist_dict_dims['dim2'])/3


def get_samples_low_high_pos_dist(df_pair_dist, low=True, n=5):
    # get top n closest/furthest proteins

    # convert to matrix for faster computation
    mat_pair_dist = df_pair_dist.values

    # an array that stores each label in its corresponding index position in sim matrix
    index_labels = df_pair_dist.index.to_numpy()

    # get the top n indexes with highest/lowest similarity for each row in the matrix (i.e the 'reduced' matrix)
    # first element is always 0 (similarity with itself), so we remove it
    if low:
        extreme_similarity_idxs = np.argsort(mat_pair_dist, axis=1)[:, 1:n + 1]
    else:
        extreme_similarity_idxs = np.argsort(mat_pair_dist, axis=1)[:, -n:]

    # get the labels of the highest similarity items
    extreme_similarity_labels = index_labels[extreme_similarity_idxs]

    # get the actual similarity values for each row
    extreme_similarity_values = np.take_along_axis(mat_pair_dist, extreme_similarity_idxs, axis=1)

    # convert into a df
    data = {
        'pdb_id1': np.repeat(index_labels, n),
        'pdb_id2': extreme_similarity_labels.flatten(),
        'distance': extreme_similarity_values.flatten()
    }
    df = pd.DataFrame(data)
    return df


def compare_pdb_pairwise_dist(pdb_id1, pdb_id2, dict_dist_no_aggr, dict_dist_aggr_dim, dist_aggr_global):
    print(f'Distances between {pdb_id1} and {pdb_id2} \n')

    print(f'No aggregation')
    for k, v in zip(dict_dist_no_aggr.keys(), dict_dist_no_aggr.values()):
        print(f'{k}:{v.loc[pdb_id1][pdb_id2]}')

    print(f'\nAggregation at dimension level')
    for k, v in zip(dict_dist_aggr_dim.keys(), dict_dist_aggr_dim.values()):
        print(f'{k}:{v.loc[pdb_id1][pdb_id2]}')

    print(f'\nGlobal aggregation')
    print(f'dim_all:{dist_aggr_global.loc[pdb_id1][pdb_id2]}')

    def get_similar_protein_pairs(df_tda_enc, pair_dist, n=5):
        # Get top pairwise distance
        df_pair_dist_top = get_samples_low_high_pos_dist(pair_dist, n=n)
        # Remove duplicates (distance is symmetric)
        df_pair_dist_top_no_dup = df_pair_dist_top.loc[
            df_pair_dist_top[['pdb_id1', 'pdb_id2']].apply(set, axis=1).drop_duplicates().index].reset_index(drop=True)
        df_pair_dist_top_no_dup = df_pair_dist_top_no_dup[
            df_pair_dist_top_no_dup['pdb_id1'] != df_pair_dist_top_no_dup['pdb_id2']]
        # Filter to proteins having different types (it's more interesting)
        df_pair_dist_top_no_dup = df_pair_dist_top_no_dup \
            .merge(df_tda_enc[['pdb_id', 'a00_dim1', 'a00_dim2']], left_on='pdb_id1', right_on='pdb_id') \
            .rename(columns={'a00_dim1': 'area_dim1_land1_pdb1', 'a00_dim2': 'area_dim2_land1_pdb1'}) \
            .drop(columns='pdb_id') \
            .merge(df_tda_enc[['pdb_id', 'a00_dim1', 'a00_dim2']], left_on='pdb_id2', right_on='pdb_id') \
            .rename(columns={'a00_dim1': 'area_dim1_land1_pdb2', 'a00_dim2': 'area_dim2_land1_pdb2'}) \
            .drop(columns='pdb_id')
        df_similar_dist = df_pair_dist_top_no_dup[df_pair_dist_top_no_dup['pdb_id1'].apply(lambda x: x[:2]) !=
                                                  df_pair_dist_top_no_dup['pdb_id2'].apply(lambda x: x[:2])]
        df_similar_dist = df_similar_dist.sort_values(by='distance').reset_index(drop=True)
        return df_similar_dist



