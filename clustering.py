import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


def cluster_data(data, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    label = clusterer.fit_predict(data)
    df_clust = pd.DataFrame(label, index=data.index, columns=['label'])
    df_clust_list = (df_clust.reset_index().groupby('label').aggregate(lambda x: x.unique().tolist()).reset_index().
                     rename(columns={'index': 'pdb_ids'}))
    return df_clust_list, label


def silhouette_analysis(dissim_mat, label):
    n_clusters = len(set(label))

    # Create figure
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.3, 1])

    # Insert blank space between silhouette plots of individual clusters
    ax1.set_ylim([0, len(dissim_mat) + (n_clusters + 1) * 10])

    # Compute average silhouette score
    silhouette_avg = silhouette_score(dissim_mat, label)
    print("Average Silhouette score :", silhouette_avg)

    # Compute the silhouette score for each sample
    sil_vals = silhouette_samples(dissim_mat, label)

    # Plot silhouette
    y_lower = 10
    for i in range(n_clusters):
        cluster_sil_vals = sil_vals[label == i]
        cluster_sil_vals.sort()

        cluster_size = cluster_sil_vals.shape[0]
        y_upper = y_lower + cluster_size

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_sil_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * cluster_size, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_title("Silhouette plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # Add vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(
        "Silhouette analysis with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    plt.show()


def remove_noise_observations_clustering(df_pairwise_dissim, label):
    # remove noise observations belonging to single element cluster (hdbscan implementation labels all noise as cluster
    # -1)
    non_noise_samples_idx = np.where(label != -1)[0]
    non_noise_pdb_idx = df_pairwise_dissim.columns.values[non_noise_samples_idx]
    df_pairwise_dissim_no_noise = pd.DataFrame(
        df_pairwise_dissim.values[non_noise_samples_idx][:, non_noise_samples_idx], index=non_noise_pdb_idx,
        columns=non_noise_pdb_idx)
    label_no_noise = label[non_noise_samples_idx]
    return df_pairwise_dissim_no_noise, label_no_noise


def analyze_clustering(df_pairwise_dissim, min_cluster_size=5, min_samples=None):
    df_cluster, lab = cluster_data(data=df_pairwise_dissim, min_cluster_size=min_cluster_size, min_samples=min_samples)
    # Do silhouette analysis (elbow? tradeoff between adding noise and increasing score accuracy)
    df_pair_l1_no_noise, lab_no_noise = remove_noise_observations_clustering(df_pairwise_dissim=df_pairwise_dissim,
                                                                             label=lab)
    print(f"num clustered samples: {len(lab_no_noise)}")
    print(f"num noise samples: {len(lab) - len(lab_no_noise)}")
    silhouette_analysis(dissim_mat=df_pair_l1_no_noise, label=lab_no_noise)
    return df_cluster
