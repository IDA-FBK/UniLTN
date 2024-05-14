import os

import ltn
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf


def plot_clusters(clst_ids, cluster_data, results_path=None):
    """
    Plot clusters based on cluster IDs and data.

    Parameters:
    - clst_ids (list): List of cluster IDs to plot.
    - cluster_data (dict): Dictionary where keys are cluster IDs and values are numpy arrays containing data points for each cluster.
    - results_path (str, optional): Path to save the plot as an image. If None, the plot will be displayed interactively.

    Returns:
    - None
    """
    for i in clst_ids:
        plt.scatter(cluster_data[i][:, 0], cluster_data[i][:, 1])

    if results_path is not None:
        plt.savefig(results_path+"clusters.png")
    else:
        plt.show()


def plot_clustering_results(data, nr_of_clusters, clst_ids, cluster_data, C, results_path):
    x0 = data[:, 0]
    x1 = data[:, 1]

    prC = [C.model([data, tf.constant([[i]] * len(data))]) for i in clst_ids]
    n = 2
    m = (nr_of_clusters + 1) // n + 1

    fig = plt.figure(figsize=(10, m * 3))

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    ax = plt.subplot2grid((m, 8), (0, 2), colspan=4)
    ax.set_title("groundtruth")
    for i in clst_ids:
        ax.scatter(cluster_data[i][:, 0], cluster_data[i][:, 1])
    for i in clst_ids:
        fig.add_subplot(m, n, i + 3)
        plt.title("C" + str(i) + "(x)")
        plt.scatter(x0, x1, c=prC[i], vmin=0, vmax=1)
        plt.colorbar()

    if results_path is not None:
        plt.savefig(results_path + "clustering_results.png")
    else:
        plt.show()


def plot_parents(parents, results_path=None):
    """
    Plot parent relationships using a directed graph.

    Parameters:
    - parents (dict): Dictionary representing parent relationships, where keys are child nodes and values are lists of parent nodes.
    - results_path (str, optional): Path to save the plot as an image. If None, the plot will be displayed interactively.

    Returns:
    - None
    """
    parDG_truth = nx.DiGraph(parents)
    pos = nx.drawing.nx_agraph.graphviz_layout(parDG_truth, prog='dot')
    nx.draw(parDG_truth, pos, with_labels=True)

    if results_path is not None:
        plt.savefig(results_path+"parents.png")
    else:
        plt.show()


# Ground Truth Ancestors
def get_descendants(entity, DG):
    """
    Get all descendants of a given entity in a directed graph.

    Parameters:
    - entity: The entity for which descendants are to be retrieved.
    - DG: The directed graph containing the relationships.

    Returns:
    - list: A list of all descendants of the specified entity.
    """
    all_d = []
    direct_d = list(DG.successors(entity))
    all_d += direct_d
    for d in direct_d:
        all_d += get_descendants(d, DG)
    return all_d


def plot_ancestor(parents, entities, path_to_results):
    """
    Plot ancestor relationships between entities.

    Parameters:
    - parents (dict): Dictionary representing parent relationships, where keys are child nodes and values are lists of parent nodes.
    - entities (list): List of entities for which ancestor relationships are to be plotted.
    - path_to_results (str): Path to save the plot as an image. If None, the plot will be displayed interactively.

    Returns:
    - None
    """
    ancestors = []
    parDG_truth = nx.DiGraph(parents)

    for e in entities:
        for d in get_descendants(e, parDG_truth):
            ancestors.append((e, d))

    ancDG_truth = nx.DiGraph(ancestors)
    pos = nx.drawing.nx_agraph.graphviz_layout(parDG_truth, prog='dot')
    nx.draw(ancDG_truth, pos, with_labels=True)

    if path_to_results is not None:
        plt.savefig(path_to_results+"ancestor.png")
    else:
        plt.show()


def plt_heatmap(df, vmin=None, vmax=None):
    plt.pcolor(df, vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    plt.colorbar()


def plots_smokes_friend_cancer_facts(facts, path_to_results):

    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt_heatmap(facts["df_smokes_cancer_facts"], vmin=0, vmax=1)
    plt.subplot(132)
    plt.title("Friend(x,y) in Group 1")
    plt_heatmap(facts["df_friends_ah_facts"], vmin=0, vmax=1)
    plt.subplot(133)
    plt.title("Friend(x,y) in Group 2")
    plt_heatmap(facts["df_friends_in_facts"], vmin=0, vmax=1)

    if path_to_results is not None:
        plt.savefig(path_to_results)
    else:
        plt.show()


def visualize_embeddings(g, g1, g2, Smokes, Friends, Cancer, path_to_results):

    x = [c.tensor.numpy() for c in g.values()]
    x_norm = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(x_norm)

    var_x = ltn.Variable("x", x)
    var_x1 = ltn.Variable("x1", x)
    var_x2 = ltn.Variable("x2", x)

    plt.figure(figsize=(8, 5))
    plt.subplot(221)
    plt.scatter(pca_transformed[:len(g1.values()), 0], pca_transformed[:len(g1.values()), 1], label="Group 1")
    plt.scatter(pca_transformed[len(g1.values()):, 0], pca_transformed[len(g1.values()):, 1], label="Group 2")
    names = list(g.keys())
    for i in range(len(names)):
        plt.annotate(names[i].upper(), pca_transformed[i])
    plt.title("Embeddings")
    plt.legend()

    plt.subplot(222)
    plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=Smokes(var_x).tensor)
    plt.title("Smokes")
    plt.colorbar()

    plt.subplot(224)
    plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=Cancer(var_x).tensor)
    plt.title("Cancer")
    plt.colorbar()

    plt.subplot(223)
    plt.scatter(pca_transformed[:len(g1.values()), 0], pca_transformed[:len(g1.values()), 1], label="Group 1")
    plt.scatter(pca_transformed[len(g1.values()):, 0], pca_transformed[len(g1.values()):, 1], label="Group 2")
    res = Friends([var_x1, var_x2]).tensor.numpy()
    for i1 in range(len(x)):
        for i2 in range(i1, len(x)):
            if (names[i1] in g1 and names[i2] in g2) \
                    or (names[i1] in g2 and names[i2] in g1):
                continue
            plt.plot(
                [pca_transformed[i1, 0], pca_transformed[i2, 0]],
                [pca_transformed[i1, 1], pca_transformed[i2, 1]],
                alpha=res[i1, i2], c="black")
    plt.title("Friendships per group")
    plt.tight_layout()

    if path_to_results is not None:
        plt.savefig(path_to_results+"embeddings.png")
    else:
        plt.show()


def plot_metric_csv(example, results_dir, exp_id, metric):
    plt.figure(figsize=(12, 6))
    for model in ["ltn", "uniltn"]:

        file_path = results_dir + model + "/" + exp_id + "/results.csv"
        print("Loading:", file_path)
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            if metric == "Satisfiability":
                plt.plot(data["Epoch"], data["test_sat_kb"], label=f"{model} Test Sat KB")
                plt.plot(data["Epoch"], data["train_sat_kb"], label=f"{model} Train Sat KB", linestyle="--")
            elif metric == "Loss":
                plt.plot(data["Epoch"], data["train_loss"], label=f"{model} Train Loss", linestyle=":")
                plt.plot(data["Epoch"], data["test_loss"], label=f"{model} Test Loss", linestyle="-.")
            elif metric == "Accuracy":
                plt.plot(data["Epoch"], data["train_accuracy"], label=f"{model} Train Accuracy", linestyle=":")
                plt.plot(data["Epoch"], data["test_accuracy"], label=f"{model} Test Accuracy", linestyle="-.")
        else:
            print("File not found:", file_path)

    plt.title(f'{metric} for {example.replace("_", " ").title()}')
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    # Setting custom ticks
    if example == "multilabel":
        x_ticks = range(0, max(data["Epoch"]) + 50, 50)
        plt.xticks(x_ticks)
        plt.yticks(np.arange(0, 1.1, 0.1))
        # Ensure max_epochs and 1.0 are displayed
        plt.xlim(0, max(data["Epoch"]) + 1)
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
    elif example == "mnist_single_digit_addition":
        x_ticks = range(0, max(data["Epoch"]) + 2, 2)
        plt.xticks(x_ticks)
        plt.yticks(np.arange(0, 1.1, 0.1))
        # Ensure max_epochs and 1.0 are displayed
        plt.xlim(0, max(data["Epoch"]))
        plt.ylim(0, 1.0)
        plt.legend()

    plt.grid(True)
    plt.savefig(results_dir + "comparison_plot_{}.png".format(metric))
    print("Done - {}".format(metric))
