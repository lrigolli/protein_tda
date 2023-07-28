import numpy as np
from data_analysis_utils.proteins_similarity import get_samples_low_high_pos_dist


def flag_connected_proteins(df_tda_enc):
    # Flag proteins we are not sure if connected by taking a look at persistence in dim 0
    df_tda_enc['connected'] = np.where(((df_tda_enc.end_dim0_0 - df_tda_enc.end_dim0_1) >
                                        (df_tda_enc.end_dim0_1 - df_tda_enc.end_dim0_4)), 'Unknown', 'True')
    return df_tda_enc


def interesting_homology_filter(df_tda_enc, filter_perc=90):
    # Infer if protein is connected from 0 dim homology
    df = flag_connected_proteins(df_tda_enc)

    # Remove proteins that may not be connected
    df = df[df['connected'] == 'True']

    # Remove proteins we are not sure if having interesting dim1 and dim2 homology by removing proteins those with
    # small area under land1
    df = df[(df['a00_dim1'] > np.percentile(df['a00_dim1'], filter_perc)) |
            (df['a00_dim2'] > np.percentile(df['a00_dim2'], filter_perc))]
    return df


def get_similar_protein_pairs(df_tda_enc, pair_dist, n=5):
    # Get top pairwise distance
    df_pair_dist_top = get_samples_low_high_pos_dist(pair_dist, n=n)
    # Remove duplicates (distance is symmetric)
    df_pair_dist_top_no_dup = df_pair_dist_top.loc[df_pair_dist_top[['pdb_id1', 'pdb_id2']].apply(set, axis=1).drop_duplicates().index].reset_index(drop=True)
    df_pair_dist_top_no_dup = df_pair_dist_top_no_dup[df_pair_dist_top_no_dup['pdb_id1'] != df_pair_dist_top_no_dup['pdb_id2']]
    # Filter to proteins having different types (it's more interesting)
    df_pair_dist_top_no_dup = df_pair_dist_top_no_dup\
        .merge(df_tda_enc[['pdb_id', 'a00_dim1', 'a00_dim2']], left_on='pdb_id1', right_on='pdb_id')\
        .rename(columns={'a00_dim1': 'area_dim1_land1_pdb1', 'a00_dim2': 'area_dim2_land1_pdb1'})\
        .drop(columns='pdb_id')\
        .merge(df_tda_enc[['pdb_id', 'a00_dim1', 'a00_dim2']], left_on='pdb_id2', right_on='pdb_id')\
        .rename(columns={'a00_dim1': 'area_dim1_land1_pdb2', 'a00_dim2': 'area_dim2_land1_pdb2'})\
        .drop(columns='pdb_id')
    df_similar_dist = df_pair_dist_top_no_dup[df_pair_dist_top_no_dup['pdb_id1'].apply(lambda x: x[:2]) !=
                                              df_pair_dist_top_no_dup['pdb_id2'].apply(lambda x: x[:2])]
    df_similar_dist = df_similar_dist.sort_values(by='distance').reset_index(drop=True)
    return df_similar_dist
