import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
import webbrowser

from IPython.display import Image


def cluster_data(data, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    label = clusterer.fit_predict(data)
    df_clust = pd.DataFrame(label, index=data.index, columns=['label'])
    print("clusters size")
    print(df_clust.value_counts())
    df_clust_list = (df_clust.reset_index().groupby('label').aggregate(lambda x: x.unique().tolist()).reset_index().
                     rename(columns={'index': 'pdb_ids'}))
    return df_clust_list


def analyze_cluster(pdb_ids, display_landscapes=True, display_persistence_diagram=False, display_rcbs_web=True,
                    display_distances=False, pair_dist_dict_dim=None):
    max_el = 10
    if len(pdb_ids) > max_el:
        print(f'Clusters with many elements are hard to compare visually, only first {max_el} elements are considered.')
        pdb_ids = pdb_ids[:max_el]

    # Load landscape pictures
    if display_landscapes:
        for suffix in ['_land_dim1', '_land_dim2']:
            for pdb in pdb_ids:
                print(pdb)
                display(Image(f"output/landscapes/{pdb}{suffix}.png"))
                plt.show()

    # Load persistence diagram pictures
    if display_persistence_diagram:
        for pdb in pdb_ids:
            print(pdb)
            display(Image(f"output/persistence_diagrams/{pdb}.png"))

    # Open links to RCBS website
    if display_rcbs_web:
        # pdb_links_3d = [f"https://www.rcsb.org/3d-sequence/{pdb_id}?assemblyId=1" for pdb_id in ex_cluster]
        pdb_links_structure = [f"https://www.rcsb.org/structure/{pdb_id}?assemblyId=1" for pdb_id in pdb_ids]
        for url in pdb_links_structure:
            webbrowser.open(url)

    if display_distances:
        print("Pairwise distance dim 1")
        print(f"{pair_dist_dict_dim['dim1'][pdb_ids].loc[pdb_ids]} \n")
        print("Pairwise distance dim 2")
        print(f"{pair_dist_dict_dim['dim2'][pdb_ids].loc[pdb_ids]} \n")
    return
