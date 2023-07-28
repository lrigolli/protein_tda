import gudhi as gd
import gudhi.representations
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from data_scripts.fourier_expansion import get_fourier_coefficients
from data_scripts.get_protein_data import read_protein3d
from constants import landscapes_params_default, output_path, landscapes_path, persistence_path


# given dict with pdb id and coordinates get simplex tree
def get_protein_simplex_tree(coords_3d, diagnostic=True):
    # create Alpha complex (nerve theorem applies as for Cech,
    # fast in dim <= 3 and has much fewer simplices than rips coplex)
    simpl_complex = gd.AlphaComplex(
        points=coords_3d
    )
    # create simplex tree
    simplex_tree = simpl_complex.create_simplex_tree()

    # 3 is fine here since nothing interesting happens in higher dimension (nerve theorem implies rips complex is
    # homotopy equivalent to subset of 3d-space made of union of balls
    # and the n-th homology group of a subset of R^n is zero, see sketch proof
    # https://math.stackexchange.com/questions/4145640/n-th-homology-of-open-subset-of-mathbbrn.
    # for radius=0 we have a union of points which is closed set, but since it is zero-dim it has 0 n-dim homology)
    # check if 1 is good choice of max_len. is something happening in persistence diagram close to 1?

    if diagnostic:
        print("Num vertices:", simplex_tree.num_vertices())
        print("Num simplices:", simplex_tree.num_simplices())
        print("Simplex tree dimension:", simplex_tree.dimension())
    return simplex_tree


def encode_land(land_all, domain, dim_land, num_landscapes, num_intervals_land, noise_threshold=99, pdb_id=None):
    lands = []
    domains = []

    for i in range(num_landscapes):
        # Restrict landscape domain interval to remove noisy observations
        land_initial = land_all[0][i*(num_intervals_land+1):(i+1)*(num_intervals_land+1)]
        cumsum_land_initial = np.cumsum(land_initial)
        # relative contribution to approximate total area of region under curve.
        # we keep observations until their summed contribution to area under curve is big enough.
        cumsum_domain_land_begin = np.max(cumsum_land_initial) * (1 - noise_threshold / 100)
        cumsum_domain_land_end = np.max(cumsum_land_initial) * (noise_threshold / 100)
        num_points_cut_left = num_intervals_land + 1 - len(cumsum_land_initial[cumsum_land_initial >
                                                                               cumsum_domain_land_begin])
        num_points_cut_right = num_intervals_land + 1 - len(cumsum_land_initial[cumsum_land_initial <
                                                                                cumsum_domain_land_end])
        # add some points to allow the landscape to go to zero more smoothly
        num_points_add_left = np.min([20, num_points_cut_left])
        num_points_add_right = np.min([20, num_points_cut_right])
        num_domain_land_points = num_intervals_land + 1 - num_points_cut_left - num_points_cut_right + num_points_add_left + num_points_add_right

        land_domain = domain[num_points_cut_left - num_points_add_left:
                             num_points_cut_left - num_points_add_left + num_domain_land_points]
        domains.append(land_domain)
        # print(f'pdb_id: {pdb_id}')
        # print(f"cut left: {num_points_cut_left}")
        # print(f"cut right: {num_points_cut_right}")
        # print(f"add left: {num_points_add_left}")
        # print(f"add right: {num_points_add_right}")
        # print(f"num domain land points: {num_domain_land_points}")
        # print(f"len land domain: {len(land_domain)}")

        # restrict lands
        land = land_all[0][i*(num_intervals_land+1) + num_points_cut_left - num_points_add_left:
                           i*(num_intervals_land+1) + num_points_cut_left - num_points_add_left +
                           num_domain_land_points]
        # print(f"len land: {len(land)}")
        if len(land) > 0:
            # Force first and last element of landscape to be zero to make sure to have a periodic function
            # (condition above is necessary because some proteins may have vanishing 1 dim or 2 dim homology)
            land[0] = 0
            land[-1] = 0
        lands.append(land)

    if pdb_id is not None:
        fig = plt.figure()
        # Plot landscapes
        for i in range(num_landscapes):
            plt.plot(domains[i], lands[i])
        plt.title(f"{dim_land}-dim landscape")
        fig.savefig(f'{landscapes_path}{pdb_id}_land_dim{dim_land}.png', dpi=100)
        fig.show()
        plt.close()
    return lands, domains


def get_landscape_encoding(simplex_tree, landscapes_params, pdb_id=None):
    print("Getting persistence homology for protein:", pdb_id)
    # compute persistence homology. this can take some time
    pers = simplex_tree.persistence()
    # death_time = [pers[i][1][1] for i in range(len(pers)) if np.isinf(pers[i][1][1]) == False]
    # max_death_time = np.max(death_time)

    if pdb_id is not None:
        # store persistence diagram for diagnostic
        pers_diag_fig = gd.plot_persistence_diagram(pers)
        pers_diag_fig.figure.savefig(f'{persistence_path}{pdb_id}.png', dpi=100)

    lands_enc_dict = {}
    # In dim 0 the encoding is done by taking the k most persisting bars born at time 0 excluding one lasting forever
    # (all connected components begin at time 0 and there is one single component persisting to infinity)
    land_enc_dim0 = simplex_tree.persistence_intervals_in_dimension(0)[:, 1][-1 - landscapes_params['land0_vars']:-1]
    lands_enc_dict['dim0'] = land_enc_dim0
    feat_names_dim0 = [f'end_dim0_{i}' for i in reversed(range(len(land_enc_dim0)))]

    # Get landscapes
    lands_raw = gd.representations.Landscape(num_landscapes=max(landscapes_params['num_landscapes_dim1'],
                                                                landscapes_params['num_landscapes_dim2']),
                                             resolution=landscapes_params['num_x_intervals_land'] + 1)

    # Get dim 1 landscape
    land_enc_dim1 = lands_raw.fit_transform([simplex_tree.persistence_intervals_in_dimension(1)])
    lands_dim1, domains_dim1 = encode_land(land_all=land_enc_dim1, domain=lands_raw.grid_, dim_land=1,
                                           num_landscapes=landscapes_params['num_landscapes_dim1'],
                                           num_intervals_land=landscapes_params['num_x_intervals_land'],
                                           pdb_id=pdb_id)

    # Approximate dim 1 landscapes as Fourier series and encode them using Fourier coefficients
    land_enc_dim1 = np.array([])
    feat_names_dim1 = []
    # Normalize domain to be [0.1]
    # Rmk: we add zero start and end points to landscape to make it zero near boundaries of interval domain
    # (hence periodic)
    num_add_points_half = 10
    for i, y in enumerate(lands_dim1):
        if len(domains_dim1[i]) > 0:
            domain_dim1_start = domains_dim1[i][0]
            domain_dim1_end = domains_dim1[i][-1]
        else:
            domain_dim1_start = 0
            domain_dim1_end = 0
        domain_dim1_num_points = len(domains_dim1[i])
        start_end_land_vec = np.array([domain_dim1_start, domain_dim1_end])
        four_coeff = get_fourier_coefficients(x=np.linspace(0, 1, domain_dim1_num_points + 2*num_add_points_half),
                                              y=np.concatenate([np.zeros(num_add_points_half), y,
                                                                np.zeros(num_add_points_half)]), fourier_degree=5)
        land_enc_vec = np.array(list(four_coeff.values()))
        land_feat_names = [f'begin{i}_dim1', f'end{i}_dim1'] + [f'{el}{i}_dim1' for el in list(four_coeff.keys())]
        land_enc_dim1 = np.concatenate([land_enc_dim1, start_end_land_vec, land_enc_vec])
        feat_names_dim1 += land_feat_names
    lands_enc_dict['dim1'] = land_enc_dim1

    # Get dim 2 landscape
    land_enc_dim2 = lands_raw.fit_transform([simplex_tree.persistence_intervals_in_dimension(2)])
    lands_dim2, domains_dim2 = encode_land(land_all=land_enc_dim2, domain=lands_raw.grid_, dim_land=2,
                                           num_landscapes=landscapes_params['num_landscapes_dim2'],
                                           num_intervals_land=landscapes_params['num_x_intervals_land'],
                                           pdb_id=pdb_id)

    # Approximate dim 2 landscapes as Fourier series and encode them using Fourier coefficients
    land_enc_dim2 = np.array([])
    feat_names_dim2 = []
    # Normalize domain to be [0.1]
    num_add_points_half = 10
    for i, y in enumerate(lands_dim2):
        if len(domains_dim2[i]) > 0:
            domain_dim2_start = domains_dim2[i][0]
            domain_dim2_end = domains_dim2[i][-1]
        else:
            domain_dim2_start = 0
            domain_dim2_end = 0
        domain_dim2_num_points = len(domains_dim2[i])
        start_end_land_vec = np.array([domain_dim2_start, domain_dim2_end])
        four_coeff = get_fourier_coefficients(x=np.linspace(0, 1, domain_dim2_num_points + 2*num_add_points_half),
                                              y=np.concatenate([np.zeros(num_add_points_half), y,
                                                                np.zeros(num_add_points_half)]), fourier_degree=5)
        land_enc_vec = np.array(list(four_coeff.values()))
        land_feat_names = [f'begin{i}_dim2', f'end{i}_dim2'] + [f'{el}{i}_dim2' for el in list(four_coeff.keys())]
        land_enc_dim2 = np.concatenate([land_enc_dim2, start_end_land_vec, land_enc_vec])
        feat_names_dim2 += land_feat_names
    lands_enc_dict['dim2'] = land_enc_dim2

    # Join representations at dimension 0,1,2 to a single vector (nothing happens in dim>=3)
    land_enc_vec = np.concatenate(list(lands_enc_dict.values()))
    feat_names = feat_names_dim0 + feat_names_dim1 + feat_names_dim2

    return land_enc_vec, feat_names


def encode_proteins_singleprocess(pdb_ids, protein_coords_dict, landscapes_params):
    protein_vecs = []
    for pdb in pdb_ids:
        simplex_tree = get_protein_simplex_tree(coords_3d=protein_coords_dict[pdb])
        protein_vec, feat_names = get_landscape_encoding(simplex_tree=simplex_tree,
                                                         landscapes_params=landscapes_params,
                                                         pdb_id=pdb)
        protein_vecs.append(protein_vec)
    if len(pdb_ids) > 0:
        df_prot = pd.DataFrame(data=protein_vecs, columns=feat_names)
        df_prot['pdb_id'] = pdb_ids
    else:
        df_prot = pd.DataFrame()
    return df_prot


def landscape_helper(chunk_idx, data, protein_coords_dict, landscapes_params):
    data_chunk = data[chunk_idx]
    df_prot = encode_proteins_singleprocess(pdb_ids=data_chunk, protein_coords_dict=protein_coords_dict,
                                            landscapes_params=landscapes_params)
    return df_prot


def encode_proteins(pdb_ids, protein_coords_dict, landscapes_params):
    # Distribute the computation on multiple CPUs. Each CPU does the computation on a different data chunk
    num_workers = multiprocessing.cpu_count() - 1
    print(f"Num workers: {num_workers}")
    # Split pdb-ids in multiple lists to allow parallel computation
    x_split = np.array_split(pdb_ids, num_workers)
    x_split = [x.tolist() for x in x_split]
    fun_for_pool = partial(landscape_helper, data=x_split,
                           protein_coords_dict=protein_coords_dict,
                           landscapes_params=landscapes_params)
    pool_process = multiprocessing.Pool(num_workers)
    df_prot_list = pool_process.map(fun_for_pool, range(len(x_split)))
    pool_process.close()
    df_prot = pd.concat(df_prot_list)
    return df_prot


def read_protein3d_helper(chunk_idx, data):
    data_chunk = data[chunk_idx]
    coord_files = read_protein3d(file_paths=data_chunk)
    return coord_files


def get_encoded_proteins_from_coords(data_type):
    num_workers = multiprocessing.cpu_count() - 1
    print(f"Encode proteins using TDA")
    coord_files = [f"data/{data_type}_coord/{f}" for f in os.listdir(f'data/{data_type}_coord') if f"protein_coords_" in f]

    # Split coordinate files in multiple lists to allow parallel computation
    coord_files_split = np.array_split(coord_files, num_workers)
    coord_files_split = [x.tolist() for x in coord_files_split]
    fun_for_pool = partial(read_protein3d_helper, data=coord_files_split)
    pool_process = multiprocessing.Pool(num_workers)
    protein_coords_list = pool_process.map(fun_for_pool, range(len(coord_files_split)))
    pool_process.close()
    protein_coords = np.concatenate(protein_coords_list)
    # protein_coords = read_protein3d(coord_files, save_plot=False)

    df_prot = pd.DataFrame()
    pdb_ids_bigger_chunk = []
    coords_bigger_chunk = {}
    # Aggregate small chunks to avoid having chunks with fewer samples than the number of workers that would slow down
    # the parallel computation
    for i, coords_chunk in enumerate(protein_coords):
        pdb_ids_chunk = list(coords_chunk.keys())
        if (len(pdb_ids_bigger_chunk) + len(pdb_ids_chunk)) < num_workers:
            pdb_ids_bigger_chunk += pdb_ids_chunk
            coords_bigger_chunk = {**coords_bigger_chunk, **coords_chunk}
        else:
            df_prot_chunk = encode_proteins(pdb_ids=pdb_ids_bigger_chunk, protein_coords_dict=coords_bigger_chunk,
                                            landscapes_params=landscapes_params_default)
            df_prot = pd.concat([df_prot, df_prot_chunk])
            pdb_ids_bigger_chunk = pdb_ids_chunk
            coords_bigger_chunk = coords_chunk
    if len(pdb_ids_bigger_chunk) > 0:
        df_prot_chunk = encode_proteins(pdb_ids=pdb_ids_bigger_chunk, protein_coords_dict=coords_bigger_chunk,
                                        landscapes_params=landscapes_params_default)
        df_prot = pd.concat([df_prot, df_prot_chunk])

    df_prot = df_prot.reset_index(drop=True)
    print(f"Encoding completed")
    df_prot.to_csv(f'{output_path}tda_encoded_{data_type}.csv', index=False)
