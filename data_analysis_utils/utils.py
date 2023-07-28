from itertools import islice
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def split_dict_chunks(data, chunk_size):
    it = iter(data)
    for i in range(0, len(data), chunk_size):
        yield {k: data[k] for k in islice(it, chunk_size)}


def split_list_chunks(data, chunk_size):
    data_chunked = []
    num_chunks = len(data) // chunk_size
    if len(data) % chunk_size > 0:
        num_chunks = num_chunks + 1
    for i in range(0, num_chunks):
        data_chunked.append(data[i*chunk_size:(i+1)*chunk_size])
    return data_chunked


def define_features_var_grouping(df):
    # Take df as input and group its columns in interpretable categories

    feat_cols = [col for col in df.columns if col not in ['pdb_id', 'enzyme_type', 'uniprot_id', 'ec_num']]
    feat_cols_dim0 = [col for col in feat_cols if 'dim0' in col]
    feat_cols_dim1 = [col for col in feat_cols if 'dim1' in col]
    feat_cols_dim2 = [col for col in feat_cols if 'dim2' in col]

    feat_cols_dim1_land1_shape = [col for col in feat_cols_dim1 if col[2] == '0']
    feat_cols_dim1_land2_shape = [col for col in feat_cols_dim1 if col[2] == '1']
    feat_cols_dim1_land3_shape = [col for col in feat_cols_dim1 if col[2] == '2']
    feat_cols_dim2_land1_shape = [col for col in feat_cols_dim2 if col[2] == '0']
    feat_cols_dim2_land2_shape = [col for col in feat_cols_dim2 if col[2] == '1']
    feat_cols_dim2_land3_shape = [col for col in feat_cols_dim2 if col[2] == '2']

    feat_cols_dim1_land1_begin_end = ['begin0_dim1', 'end0_dim1']
    feat_cols_dim1_land2_begin_end = ['begin1_dim1', 'end1_dim1']
    feat_cols_dim1_land3_begin_end = ['begin2_dim1', 'end2_dim1']
    feat_cols_dim2_land1_begin_end = ['begin0_dim2', 'end0_dim2']
    feat_cols_dim2_land2_begin_end = ['begin1_dim2', 'end1_dim2']
    feat_cols_dim2_land3_begin_end = ['begin2_dim2', 'end2_dim2']

    feat_cols_begin_end =\
        feat_cols_dim1_land1_begin_end +\
        feat_cols_dim1_land2_begin_end +\
        feat_cols_dim1_land3_begin_end +\
        feat_cols_dim2_land1_begin_end +\
        feat_cols_dim2_land2_begin_end +\
        feat_cols_dim2_land3_begin_end

    feat_cols_dim1_land1 = feat_cols_dim1_land1_begin_end + feat_cols_dim1_land1_shape
    feat_cols_dim1_land2 = feat_cols_dim1_land2_begin_end + feat_cols_dim1_land2_shape
    feat_cols_dim1_land3 = feat_cols_dim1_land3_begin_end + feat_cols_dim1_land3_shape
    feat_cols_dim2_land1 = feat_cols_dim2_land1_begin_end + feat_cols_dim2_land1_shape
    feat_cols_dim2_land2 = feat_cols_dim2_land2_begin_end + feat_cols_dim2_land2_shape
    feat_cols_dim2_land3 = feat_cols_dim2_land3_begin_end + feat_cols_dim2_land3_shape

    features_var_dict_dim_land_shape_extrema = {'dim0': feat_cols_dim0,
                                                'dim1_land1_shape': feat_cols_dim1_land1_shape,
                                                'dim1_land2_shape': feat_cols_dim1_land2_shape,
                                                'dim1_land3_shape': feat_cols_dim1_land3_shape,
                                                'dim2_land1_shape': feat_cols_dim2_land1_shape,
                                                'dim2_land2_shape': feat_cols_dim2_land2_shape,
                                                'dim2_land3_shape': feat_cols_dim2_land3_shape,
                                                'dim1_land1_begin_end': feat_cols_dim1_land1_begin_end,
                                                'dim1_land2_begin_end': feat_cols_dim1_land2_begin_end,
                                                'dim1_land3_begin_end': feat_cols_dim1_land3_begin_end,
                                                'dim2_land1_begin_end': feat_cols_dim2_land1_begin_end,
                                                'dim2_land2_begin_end': feat_cols_dim2_land2_begin_end,
                                                'dim2_land3_begin_end': feat_cols_dim2_land3_begin_end}
    features_var_dict_dim_land = {'dim0': feat_cols_dim0,
                                  'dim1_land1': feat_cols_dim1_land1,
                                  'dim1_land2': feat_cols_dim1_land2,
                                  'dim1_land3': feat_cols_dim1_land3,
                                  'dim2_land1': feat_cols_dim2_land1,
                                  'dim2_land2': feat_cols_dim2_land2,
                                  'dim2_land3': feat_cols_dim2_land3}
    features_var_dict_dim = {'dim0': feat_cols_dim0, 'dim1': feat_cols_dim1, 'dim2': feat_cols_dim2}
    grouped_variables = {
        'feat_cols': feat_cols,
        'feat_cols_dim0': feat_cols_dim0,
        'feat_cols_begin_end': feat_cols_begin_end,
        'features_var_dict_dim_land_shape_extrema': features_var_dict_dim_land_shape_extrema,
        'features_var_dict_dim_land': features_var_dict_dim_land,
        'features_var_dict_dim': features_var_dict_dim}
    return grouped_variables


def outlier_removal(df, cols_outliers_check, ratio_outliers_remove=0.1):
    # identify outliers in the training dataset based on begin and end of 0,1,and 2 dim features
    # (this should hint errors in recorded coordinates and the resulting distributions looks nicer than if we remove
    # only dim0 features)
    # contamination gives the proportion of outliers we wish to remove, default is "auto"
    iso_forest = IsolationForest(contamination=ratio_outliers_remove)
    # contamination was set so that the feature distribution histograms are close to normal
    yhat = iso_forest.fit_predict(df[cols_outliers_check])
    # select all rows that are not outliers
    mask = yhat != -1
    print("num original samples:", len(df))
    print("num samples after removing outliers:", sum(mask))
    # Remove outliers
    df_no_out = df[mask]
    return df_no_out


def check_features_distribution(df, features_var_dict_dim_land):
    for key in features_var_dict_dim_land.keys():
        df[features_var_dict_dim_land[key]].hist(figsize=(10,10))
        print(key)
        plt.show()
