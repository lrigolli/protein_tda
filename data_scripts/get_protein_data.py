import os
import codecs
import json
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from data_analysis_utils.utils import split_list_chunks
from constants import plots3d_path


def download_pdb_from_id(data_type):
    print(f"Download {data_type} PDB files")
    os.system(f" ./data/batch_download.sh -f data/pdb_ids/{data_type}.txt -p")
    pdb_files = [file for file in os.listdir() if '.pdb' in file]
    os.makedirs(name=f"data/{data_type}", exist_ok=True)
    for file in pdb_files:
        os.replace(file, f"data/{data_type}/{file}")


def extract_3dcoords_singleprocess(zipped_files_list, files_parent_dir, output_coord_filename):
    pdb_coord_dict = {}
    for file_name in zipped_files_list:
        os.system(f"gzip -dk {files_parent_dir}{file_name}")
        pdb_file = file_name.split('.gz')[0]
        pdb_name = file_name.split('.pdb')[0]
        try:
            traj = md.load_pdb(f"{files_parent_dir}{file_name}")
            pdb_coord_dict[pdb_name] = traj.xyz[0].tolist()
        except:
            ""
        finally:
            ""
        if os.path.exists(f"{files_parent_dir}{pdb_file}"):
            os.remove(f"{files_parent_dir}{pdb_file}")
        json.dump(pdb_coord_dict, codecs.open(output_coord_filename, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)


def extract_3dcoords_helper(chunk_idx, data, files_parent_dir, output_coord_filename):
    print(f"data chunk: {chunk_idx}")
    data_chunk = data[chunk_idx]
    extract_3dcoords_singleprocess(zipped_files_list=data_chunk,
                                   files_parent_dir=files_parent_dir,
                                   output_coord_filename=f"{output_coord_filename}_{chunk_idx}.json")


def extract_3dcoords_from_pdb(data_type):
    path_dir = f"data/{data_type}/"
    # Make sure there are no unzipped files
    if os.path.exists(f"{path_dir}*.pdb"):
        os.remove(f"{path_dir}*.pdb")
    pdb_zipped_files = [file for file in np.sort(os.listdir(path_dir)) if '.pdb.gz' in file]

    # Create a directory to store coordinates
    path_dir_coord = f"data/{data_type}_coord"
    os.makedirs(name=path_dir_coord, exist_ok=True)

    # Distribute the computation on multiple CPUs. Each CPU does the computation on a different data chunk
    num_workers = multiprocessing.cpu_count() - 1

    # Split pdb-ids in multiple lists to allow parallel computation
    x_split = split_list_chunks(data=pdb_zipped_files, chunk_size=20)

    # Multprocessing call
    print(f"Store PDB coordinates as json files")
    fun_for_pool = partial(extract_3dcoords_helper,
                           data=x_split,
                           files_parent_dir=path_dir,
                           output_coord_filename=f'{path_dir_coord}/protein_coords')
    pool_process = multiprocessing.Pool(num_workers)
    pool_process.map(fun_for_pool, range(len(x_split)))
    pool_process.close()


def read_protein3d(file_paths, save_plot=True):
    coord_files = []
    print("Reading protein 3d coordinates...")
    if save_plot:
        print("Saving option enabled: this can take some time")
    # Read from list of file paths
    for path in file_paths:
        print(f"Reading path: {path}")
        # Read json file with 3d coordinates of protein atoms
        data_utf = codecs.open(path, 'r', encoding='utf-8').read()
        data_json = json.loads(data_utf)
        for key in data_json.keys():
            data_json[key] = np.array(data_json[key])
            if save_plot:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(data_json[key][:, 0], data_json[key][:, 1], data_json[key][:, 2], alpha=0.5, s=2)
                ax.set_title(f'{key} ({len(data_json[key])} atoms)')
                fig.savefig(f'{plots3d_path}{key}.png', dpi=50)
                plt.close()
        coord_files.append(data_json)
    return coord_files


"""def read_pdb_id_from_coord_file(file_paths):
    pdb_ids = []
    for path in file_paths:
        # Read pdb ids from json file with enzyme coordinates
        data_str = codecs.open(path, 'r', encoding='utf-8').read()
        data_json = json.loads(data_str)
        pdb_ids_chunk = list(data_json.keys())
        pdb_ids += pdb_ids_chunk
    return pdb_ids
"""