from glob import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def threshold_till_dag(B):
    """Remove the edges with smallest absolute weight until a DAG is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
        float: Minimum threshold to obtain DAG.
    """
    if is_dag(B):
        return B, 0

    B = np.copy(B)
    # Get the indices with non-zero weight
    nonzero_indices = np.where(B != 0)
    # Each element in the list is a tuple (weight, j, i)
    weight_indices_ls = list(zip(B[nonzero_indices],
                                 nonzero_indices[0],
                                 nonzero_indices[1]))
    # Sort based on absolute weight
    sorted_weight_indices_ls = sorted(weight_indices_ls, key=lambda tup: abs(tup[0]))

    for weight, j, i in sorted_weight_indices_ls:
        if is_dag(B):
            # A DAG is found
            break

        # Remove edge with smallest absolute weight
        B[j, i] = 0
        dag_thres = abs(weight)

    return B, dag_thres


def postprocess(B, graph_thres=0.3):
    """Post-process estimated solution:
        (1) Thresholding.
        (2) Remove the edges with smallest absolute weight until a DAG
            is obtained.

    Args:
        B (numpy.ndarray): [d, d] weighted matrix.
        graph_thres (float): Threshold for weighted matrix. Default: 0.3.

    Returns:
        numpy.ndarray: [d, d] weighted matrix of DAG.
    """
    B = np.copy(B)
    B[np.abs(B) <= graph_thres] = 0    # Thresholding
    B, _ = threshold_till_dag(B)

    return B


def checkpoint_after_training(output_dir, X, B_true, B_init,
                              B_est, B_processed, print_func):
    """Checkpointing after the training ends.

    Args:
        output_dir (str): Output directory to save training outputs.
        X (numpy.ndarray): [n, d] data matrix.
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_init (numpy.ndarray or None): [d, d] weighted matrix for
            initialization. Set to None to disable. Default: None.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        print_func (function): Printing function.
    """
    # Visualization
    plot_solution(B_true, B_est, B_processed,
                  save_name='{}/plot_solution.jpg'.format(output_dir))
    print_func("Finished plotting estimated graph (without post-processing).")

    results = count_accuracy(B_true != 0, B_processed != 0)
    print_func("Results (after post-processing): {}.".format(results))

    # Save training outputs
    np.save('{}/X.npy'.format(output_dir), X)
    np.save('{}/B_true.npy'.format(output_dir), B_true)
    np.save('{}/B_est.npy'.format(output_dir), B_est)
    np.save('{}/B_processed.npy'.format(output_dir), B_processed)
    if B_init is not None:
        np.save('{}/B_init.npy'.format(output_dir), B_init)
    print_func("Finished saving training outputs at {}.".format(output_dir))


def set_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def count_accuracy(B_bin_true, B_bin_est, check_input=False):
    """Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1}, 
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code modified from:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'pred_size': pred_size}


def plot_solution(B_true, B_est, B_processed, save_name=None):
    """Checkpointing after the training ends.

    Args:
        B_true (numpy.ndarray): [d, d] weighted matrix of ground truth.
        B_est (numpy.ndarray): [d, d] estimated weighted matrix.
        B_processed (numpy.ndarray): [d, d] post-processed weighted matrix.
        save_name (str or None): Filename to solve the plot. Set to None
            to disable. Default: None.
    """
    fig, axes = plt.subplots(figsize=(10, 3), ncols=3)

    # Plot ground truth
    im = axes[0].imshow(B_true, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[0].set_title("Ground truth", fontsize=13)
    axes[0].tick_params(labelsize=13)

    # Plot estimated solution
    im = axes[1].imshow(B_est, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[1].set_title("Estimated solution", fontsize=13)
    axes[1].set_yticklabels([])    # Remove yticks
    axes[1].tick_params(labelsize=13)

    # Plot post-processed solution
    im = axes[2].imshow(B_processed, cmap='RdBu', interpolation='none',
                        vmin=-2.25, vmax=2.25)
    axes[2].set_title("Post-processed solution", fontsize=13)
    axes[2].set_yticklabels([])    # Remove yticks
    axes[2].tick_params(labelsize=13)

    # Adjust space between subplots
    fig.subplots_adjust(wspace=0.1)

    # Colorbar (with abit of hard-coding)
    im_ratio = 3 / 10
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.05*im_ratio, pad=0.035)
    cbar.ax.tick_params(labelsize=13)
    # plt.show()

    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')