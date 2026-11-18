import numpy as np


def distance_logp(dist_m, sigma):
    """Gaussian log-likelihood of the GPS-to-arc distance: GPS noise is roughly
    zero-mean Gaussian around the true road position."""
    d = np.asarray(dist_m, float)
    return -0.5 * (d / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))


def heading_logp(heading_diff_deg, sigma_deg):
    """Gaussian penalty on the angle between GPS heading and the DIRECTED arc bearing.
    Twin arcs share geometry, so distance cannot separate them; heading can (0 vs 180).
    A neutral 90 (slow points) is the same for both twins, so it cannot mislead."""
    h = np.asarray(heading_diff_deg, float)
    return -0.5 * (h / sigma_deg) ** 2
