import numpy as np


def flag_connected_proteins(df_tda_enc, max_perc_threshold=70):
    # Flag proteins we are not sure if connected by taking a look at persistence in dim 0
    max_perc_connected = np.percentile(df_tda_enc['end_dim0_0'], max_perc_threshold)
    df_tda_enc['connected'] = np.where((df_tda_enc.end_dim0_0 < max_perc_connected), 'Unknown', 'True')
    return df_tda_enc


def interesting_homology_filter(df_tda_enc, filter_perc=80):
    # Infer if protein is connected from 0 dim homology
    df = flag_connected_proteins(df_tda_enc)

    # Remove proteins that may not be connected
    df = df[df['connected'] == 'True']

    # Remove proteins we are not sure if having interesting dim1 and dim2 homology by removing proteins those with
    # small area under land1
    df = df[(df['a00_dim1'] > np.percentile(df['a00_dim1'], filter_perc)) |
            (df['a00_dim2'] > np.percentile(df['a00_dim2'], filter_perc))]
    return df
