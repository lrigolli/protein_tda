from data_scripts.get_protein_data import download_pdb_from_id, extract_3dcoords_from_pdb
from data_scripts.tda_encoding import get_encoded_proteins_from_coords


def tda_encoding_pipeline(data_type='proteins_assembly'):
    # 1) Download PDB files corresponding to IDs in pdb_ids/proteins_assembly.txt file
    #download_pdb_from_id(data_type=data_type)

    # 2) Extract 3d coordinates from PDB files and store them as json (key=PDB_ID, value=3d_coord)
    # extract_3dcoords_from_pdb(data_type=data_type)

    # 3) Encode protein 3d coordinates using TDA
    get_encoded_proteins_from_coords(data_type=data_type)
