import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from IPython.display import Image, display
from constants import plots3d_path, persistence_path, landscapes_path


def compare_pdb_plots(pdb_list):
    for path_dir in [plots3d_path, persistence_path]:
        for pdb_id in pdb_list:
            display(Image(filename=f'{path_dir}{pdb_id}.png'))
    for pdb_id in pdb_list:
        display(Image(filename=f'{landscapes_path}{pdb_id}_land_dim1.png'))
    for pdb_id in pdb_list:
        display(Image(filename=f'{landscapes_path}{pdb_id}_land_dim2.png'))


def apply_mds(title, df_dissim, labels=None, enzyme_color_dict=None, n_comp=3):
    embedding = MDS(n_components=n_comp, dissimilarity='precomputed', normalized_stress='auto')
    x_emb = embedding.fit_transform(df_dissim.values)
    fig = plt.figure(figsize=(12, 12))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    if (labels is not None) and (enzyme_color_dict is not None):
        ax.scatter(x_emb[:, 0], x_emb[:, 1], x_emb[:, 2], s=40, c=labels.map(enzyme_color_dict), marker='o', alpha=0.5)
    else:
        ax.scatter(x_emb[:, 0], x_emb[:, 1], x_emb[:, 2], s=40, marker='o', alpha=0.5)
    ax.set_title(f'{title}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if (labels is not None) and (enzyme_color_dict is not None):
        handles = [mpatches.Patch(color=item) for item in enzyme_color_dict.values()]
        label = enzyme_color_dict.keys()
        plt.legend(handles, label, loc='upper right', prop={'size': 10})
    else:
        plt.legend(loc='upper right', prop={'size': 10})
    plt.show()


""" 
# TSNE
#tsne_result = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(x_train_no_out)
# Plot the result of our TSNE with the label color coded
# A lot of the stuff here is about making the plot look pretty and not TSNE
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_train_no_out})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)"""

"""# PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x_train_no_out)
# Apply PCA for data visualization
# do not scale!
pca = PCA(n_components = 2)
data_pca = pca.fit_transform(x_train_no_out)
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2'])
data_pca['enzyme_type'] = y_train_no_out
sns.scatterplot(data=data_pca, x="PC1", y="PC2", hue="enzyme_type",alpha=0.7)"""