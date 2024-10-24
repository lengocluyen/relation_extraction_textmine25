import logging
import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    assert y_true.shape == y_pred.shape
    Y = np.c_[y_true, (y_true.sum(axis=1) == 0).astype(int)]
    Z = np.c_[y_pred, (y_pred.sum(axis=1) == 0).astype(int)]
    q = Y.shape[1]  # number of labels
    N = Y.shape[0]  # number of examples

    M = np.zeros(shape=(q, q))
    for i in range(N):
        if all(Y[i] == Z[i]):
            logging.debug(f"{i} case i")
            M += np.diag(Y[i])
        else:
            yi = set(np.argwhere(Y[i] > 0).flatten().tolist())
            zi = set(np.argwhere(Z[i] > 0).flatten().tolist())
            yi_diff_zi_arr = np.zeros(q)
            yi_diff_zi_arr[list(yi.difference(zi))] = 1
            zi_diff_yi_arr = np.zeros(q)
            zi_diff_yi_arr[list(zi.difference(yi))] = 1
            yi_inter_zi_arr = np.zeros(q)
            yi_inter_zi_arr[list(yi.intersection(zi))] = 1
            if len(yi.difference(zi)) == 0 and len(zi.difference(yi)) > 0:
                logging.debug(f"{i} case ii")
                M += (
                    np.outer(yi_inter_zi_arr, zi_diff_yi_arr) + len(yi) * np.diag(Y[i])
                ) / len(zi)
            elif len(yi.difference(zi)) > 0 and len(zi.difference(yi)) == 0:
                logging.debug(f"{i} case iii")
                M += np.outer(yi_diff_zi_arr, Z[i]) / len(zi) + np.diag(Z[i])
            else:
                logging.debug(f"{i} case iv")
                M += np.outer(yi_diff_zi_arr, zi_diff_yi_arr) / len(
                    zi.difference(yi)
                ) + np.diag(yi_inter_zi_arr)
    return M
