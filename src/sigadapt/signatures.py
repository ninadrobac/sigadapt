"""Sliding-window signatures with time and basepoint augmentation.
"""

from itertools import product

import iisignature
import numpy as np


def sig_sliding_basepoint(covariates, window, trunc_order=4, drop=False, recompute_every=500):
    """

    Parameters
    ----------
    covariates : array of shape (n,) or (n, c)
        Components of the path (time is added as the 0th).
    window : int
        Window length, in number of increments.
    trunc_order : int
        Truncation order of the signature.
    drop : bool
        If True, drop the pure-time components (0), (0,0), ... of each level.
    recompute_every : int
        Recompute the signature from scratch every 'recompute_every' updates.

    Returns
    -------
    array of shape (n, siglength)
        Row t is the signature of the window ending at index t (rows t < window are zero).
    """
    covariates = np.asarray(covariates)
    if covariates.ndim == 1:
        covariates = covariates.reshape(-1, 1)

    n_samples = covariates.shape[0]
    dim = covariates.shape[1] + 1
    n = window + 1

    sig_matrix = np.zeros((n_samples, iisignature.siglength(dim, trunc_order)))

    # First window, with the basepoint (0, 0) in front
    time = (np.arange(window + 1) + 1) / n
    path = np.column_stack((time, covariates[: window + 1]))
    path = np.vstack([np.zeros((1, dim)), path])

    current_sig = iisignature.sig(path, trunc_order)
    sig_matrix[window] = current_sig
    time_stamp = n

    for i in range(1, n_samples - window):
        # Remove the two first segments 
        old_segment = path[0:3][::-1]
        current_sig = iisignature.sigcombine(iisignature.sig(old_segment, trunc_order),
                                             current_sig, dim, trunc_order)

        # Append the new observation
        time_stamp += 1
        new_point = np.concatenate(([time_stamp / n], covariates[i + window]))
        path = np.vstack([path, new_point])

        # Move the basepoint forward in time and remove the oldest observation
        path[0, 0] += 1 / n
        path = np.delete(path, 1, axis=0)

        # Add the new first segment 
        current_sig = iisignature.sigcombine(iisignature.sig(path[0:2], trunc_order),
                                             current_sig, dim, trunc_order)

        # Add the new last segment
        current_sig = iisignature.sigcombine(current_sig, iisignature.sig(path[-2:], trunc_order),
                                             dim, trunc_order)

        # Periodic recomputation on the current window
        if recompute_every and i % recompute_every == 0:
            t0 = path[0, 0]
            path[:, 0] -= t0
            time_stamp = n
            current_sig = iisignature.sig(path, trunc_order)

        sig_matrix[window + i] = current_sig

    if drop:
        powers = [dim**k for k in range(1, trunc_order)]
        indices = np.insert(np.cumsum(powers), 0, 0)
        sig_matrix = np.delete(sig_matrix, indices, axis=1)

    return sig_matrix


def signature_words(dim, depth):
    """ Words indexing the signature coordinates, in in lexicographic order """
    return [w for k in range(1, depth + 1) for w in product(range(dim), repeat=k)]


def keep_mask(words, drop_pure=(), require_any=()):
    """ Boolean mask over 'words'.

    drop_pure   : letters l such that the pure words (l), (l,l), ... are removed.
    require_any : if non-empty, keep only the words containing at least one of these letters.
    """
    keep = np.ones(len(words), dtype=bool)
    for i, w in enumerate(words):
        letters = set(w)
        if len(letters) == 1 and next(iter(letters)) in drop_pure:
            keep[i] = False
        if require_any and not letters & set(require_any):
            keep[i] = False
    return keep
