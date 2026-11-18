"""Log-space Viterbi with back-pointers.

delta_t(j) = e_t(j) + max_i [ delta_{t-1}(i) + A_t(i, j) ],  delta_0(j) = e_0(j)
Log space turns products of many small probabilities into sums, so a long trip
(thousands of points) does not underflow to 0.0. Cost is O(T * K^2).
"""
from typing import Callable

import numpy as np


def viterbi(emission_logp: list[np.ndarray],
            transition_logp: Callable[[int], np.ndarray]) -> list[int]:
    """Best candidate index per step.

    emission_logp[t] has shape (K_t,); transition_logp(t) returns the (K_{t-1}, K_t)
    log-probability matrix for the step t-1 -> t (t >= 1). K may change per step.
    """
    T = len(emission_logp)
    if T == 0:
        return []
    delta = np.asarray(emission_logp[0], float)
    back: list[np.ndarray] = []
    for t in range(1, T):
        scores = delta[:, None] + transition_logp(t)  # (K_prev, K_cur)
        bp = np.argmax(scores, axis=0)
        delta = scores[bp, np.arange(scores.shape[1])] + emission_logp[t]
        back.append(bp)
    path = [int(np.argmax(delta))]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return path[::-1]
